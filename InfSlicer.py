import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import torch
from ultralytics import YOLO
import supervision as sv  # pip install supervision

# =============================================================================
#  CONFIGURATION & GLOBAL STATE
# =============================================================================
MODEL_PATH = "cbb.pt"

# Reference design resolution — all coordinates below were drawn for this.
# At runtime they are scaled to whatever screen the app opens on.
REF_W, REF_H = 1280, 720

cap = cv2.VideoCapture(0) # Adjust index (0, 1, or 2) if camera doesn't show
is_paused = False
last_frame = None
camera_label = None
results_label = None
model = None

# Layout globals — populated in start_app() and used by display_on_label()
SCREEN_W = REF_W
SCREEN_H = REF_H
PREVIEW_W = 850
PREVIEW_H = 595

# --- Load YOLO Model ---
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL_PATH)
    model.to(device)
    print(f"✅ Model loaded on {device}")
except Exception as e:
    print(f"❌ Model Load Error: {e}")

# =============================================================================
#  INFERENCE SLICER PARAMETERS  ← MODIFY THESE AS NEEDED
# =============================================================================
#
#  model_path               : Path to the .pt model used by the Inference Slicer.
#  slice_height/width       : Pixel size of each tile. Smaller = detects tinier objects.
#  overlap_height/width_ratio: Tile overlap ratio (0.0–1.0). Higher = fewer border misses.
#  model_confidence_threshold: Min confidence per detection inside each slice.
#  certainty_threshold      : Post-merge filter — detections BELOW this are discarded.
#  iou_threshold            : IoU used when merging duplicate detections across tiles.
#  postprocess_class_agnostic: If True, NMS merges boxes across all classes.

SAHI_CONFIG = {
    "model_path": "cbb.pt",                 # ← model used for Inference Slicer
    "slice_height": 256,                    # ← tile height in pixels
    "slice_width": 256,                     # ← tile width in pixels
    "overlap_height_ratio": 0.4,            # ← vertical overlap between tiles
    "overlap_width_ratio": 0.4,             # ← horizontal overlap between tiles
    "model_confidence_threshold": 0.3,      # ← per-slice detection confidence floor
    "certainty_threshold": 0.59,            # ← post-merge confidence filter
    "iou_threshold": 0.5,                   # ← IoU for merging cross-tile duplicates
    "postprocess_class_agnostic": True      # ← class-agnostic NMS (noted, not native to sv)
}

# --- Load YOLO model for Inference Slicer ---
slicer_yolo_model = None
try:
    slicer_yolo_model = YOLO(SAHI_CONFIG["model_path"])  # ← SAHI_CONFIG["model_path"]
    slicer_yolo_model.to(device)
    print(f"✅ Slicer model loaded: {SAHI_CONFIG['model_path']}")
except Exception as e:
    print(f"❌ Slicer Model Load Error: {e}")

# =============================================================================
#  CLASSIFICATION HELPER  (shared by both inference paths)
# =============================================================================
# BGR colors used for the bounding boxes (OpenCV uses BGR, not RGB)
COLOR_INFESTED     = (0, 0, 255)    # Red
COLOR_NON_INFESTED = (0, 255, 0)    # Green
COLOR_UNKNOWN      = (0, 255, 255)  # Yellow


def classify_detection(label_name, score):
    """Map a raw model label + confidence to (display_label, BGR_color).

    - Low-confidence detections fall through to "Unknown" (yellow).
    - Anything matching non/healthy/clean keywords becomes Non-Infested (green).
    - Anything matching infest/borer/cbb keywords becomes Infested (red).
    - Unrecognized class names default to Unknown (yellow).
    The non-infested check runs BEFORE the infested check, otherwise a label
    like "non-infested" would match the substring "infest" and turn red.
    """
    if score < SAHI_CONFIG["certainty_threshold"]:
        return "Unknown", COLOR_UNKNOWN

    lname = label_name.lower().strip()

    if any(k in lname for k in ("non", "healthy", "clean", "good")):
        return label_name, COLOR_NON_INFESTED

    if any(k in lname for k in ("infest", "borer", "cbb", "damaged", "bad")):
        return label_name, COLOR_INFESTED

    return "Unknown", COLOR_UNKNOWN


# =============================================================================
#  DETECTION LOGIC
# =============================================================================
def run_yolo_inference(frame):
    """Runs standard YOLO and returns (annotated_frame, tally_dict)"""
    if model is None: return frame, {}
    
    annotated_img = frame.copy()
    label_counts = {}
    
    results = model(annotated_img, iou=0.5, conf=0.3, verbose=False)
    
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # Data extraction
            xyxy = box.xyxy[0].int().tolist()
            cls_id = int(box.cls[0].item())
            raw_label = model.names[cls_id]
            score = float(box.conf[0].item()) if hasattr(box, "conf") else 1.0

            # Classify and pick color
            display_label, color = classify_detection(raw_label, score)

            # Tally update
            label_counts[display_label] = label_counts.get(display_label, 0) + 1

            # Drawing
            cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(annotated_img, f"{display_label} {score:.2f}", (xyxy[0], xyxy[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    return annotated_img, label_counts


def run_roboflow_slicer_inference(frame):
    """
    Runs Roboflow (supervision) InferenceSlicer and returns (annotated_frame, tally_dict).
    All parameters drawn from SAHI_CONFIG above.
    """
    if slicer_yolo_model is None: return frame, {}

    annotated_img = frame.copy()
    label_counts = {}

    # --- Callback: runs YOLO on each tile ---
    def slicer_callback(image_slice):
        results = slicer_yolo_model(
            image_slice,
            conf=SAHI_CONFIG["model_confidence_threshold"],              # ← SAHI_CONFIG["model_confidence_threshold"]
            iou=SAHI_CONFIG["iou_threshold"],                            # ← SAHI_CONFIG["iou_threshold"]
            verbose=False,
        )
        return sv.Detections.from_ultralytics(results[0])

    # --- Convert overlap ratios to pixel values (compatible with older supervision versions) ---
    overlap_w_px = int(SAHI_CONFIG["slice_width"]  * SAHI_CONFIG["overlap_width_ratio"])   # ← SAHI_CONFIG["slice_width"]  * SAHI_CONFIG["overlap_width_ratio"]
    overlap_h_px = int(SAHI_CONFIG["slice_height"] * SAHI_CONFIG["overlap_height_ratio"])  # ← SAHI_CONFIG["slice_height"] * SAHI_CONFIG["overlap_height_ratio"]

    # --- Build slicer using config ---
    slicer = sv.InferenceSlicer(
        callback=slicer_callback,
        slice_wh=(SAHI_CONFIG["slice_width"], SAHI_CONFIG["slice_height"]),  # ← SAHI_CONFIG["slice_width/height"]
        overlap_wh=(overlap_w_px, overlap_h_px),                             # ← derived from SAHI_CONFIG["overlap_*_ratio"]
        iou_threshold=SAHI_CONFIG["iou_threshold"],                          # ← SAHI_CONFIG["iou_threshold"]
    )

    # --- Run slicer on the full frame ---
    detections = slicer(frame)

    for i in range(len(detections)):
        score = float(detections.confidence[i]) if detections.confidence is not None else 1.0

        xyxy = detections.xyxy[i].astype(int).tolist()
        cls_id = int(detections.class_id[i])
        raw_label = slicer_yolo_model.names[cls_id]

        # Classify and pick color (low-confidence falls through to Unknown/yellow)
        display_label, color = classify_detection(raw_label, score)

        # Tally update
        label_counts[display_label] = label_counts.get(display_label, 0) + 1

        # Drawing
        cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        cv2.putText(annotated_img, f"{display_label} {score:.2f}", (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated_img, label_counts

# =============================================================================
#  GUI COMPONENT LOGIC
# =============================================================================

def display_on_label(frame):
    """Central function to update the main preview label.
    Uses PREVIEW_W / PREVIEW_H globals so it adapts to the current screen size.
    """
    global camera_label, PREVIEW_W, PREVIEW_H
    if frame is None or camera_label is None:
        return

    w = max(1, PREVIEW_W)
    h = max(1, PREVIEW_H)
    frame_resized = cv2.resize(frame, (w, h))
    cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    camera_label.imgtk = imgtk  # Keep reference to prevent flicker/blanking
    camera_label.configure(image=imgtk)

def update_camera_feed():
    """Live feed loop"""
    global is_paused, last_frame
    if is_paused: return
    
    ret, frame = cap.read()
    if ret:
        last_frame = frame
        display_on_label(frame)
    
    camera_label.after(10, update_camera_feed)

def gallery_button_clicked():
    save_folder = "processed_images"

    # Create folder silently if missing so the gallery just shows "empty"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    images = [
        f for f in os.listdir(save_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Font sizes scaled off the actual screen, not hardcoded
    base = min(SCREEN_W / REF_W, SCREEN_H / REF_H)
    f_big   = max(10, int(24 * base))
    f_med   = max(9,  int(14 * base))
    f_small = max(8,  int(11 * base))

    if not images:
        popup = tk.Toplevel()
        popup.title("Gallery")
        popup.attributes("-fullscreen", True)
        popup.configure(bg="#FFD782")
        tk.Label(popup, text="No saved images yet.", bg="#FFD782",
                 font=("Arial", f_big, "bold")).pack(expand=True)
        tk.Button(popup, text="CLOSE", bg="#C91B1A", fg="white",
                  font=("Arial", f_med, "bold"),
                  command=popup.destroy).pack(pady=int(SCREEN_H * 0.04))
        popup.bind("<Escape>", lambda e: popup.destroy())
        return

    images.sort(reverse=True)

    gal_win = tk.Toplevel()
    gal_win.attributes("-fullscreen", True)
    gal_win.configure(bg="#FFD782")
    gal_win.bind("<Escape>", lambda e: gal_win.destroy())

    # --- Top bar ---
    top_bar = tk.Frame(gal_win, bg="#FFD782")
    top_bar.pack(side="top", fill="x",
                 padx=int(SCREEN_W * 0.015), pady=int(SCREEN_H * 0.02))

    tk.Button(top_bar, text="← CLOSE", command=gal_win.destroy,
              font=("Arial", f_med, "bold"), bg="white",
              fg="#C91B1A", borderwidth=0, cursor="hand2").pack(side="left")

    counter_var = tk.StringVar()
    tk.Label(top_bar, textvariable=counter_var, bg="#FFD782",
             font=("Arial", f_med, "bold")).pack(side="right")

    # --- Image viewer ---
    viewer = tk.Label(gal_win, bg="#FFD782")
    viewer.pack(expand=True)

    # --- Filename label ---
    name_var = tk.StringVar()
    tk.Label(gal_win, textvariable=name_var, bg="#FFD782",
             font=("Arial", f_small)).pack(pady=int(SCREEN_H * 0.006))

    # --- Navigation ---
    nav_frame = tk.Frame(gal_win, bg="#FFD782")
    nav_frame.pack(side="bottom", pady=int(SCREEN_H * 0.03))

    current_idx = [0]

    # Thumbnail box: ~85% × ~75% of screen so big displays don't show tiny pics
    thumb_w = int(SCREEN_W * 0.85)
    thumb_h = int(SCREEN_H * 0.75)

    def show_image():
        filename = images[current_idx[0]]
        filepath = os.path.join(save_folder, filename)
        try:
            pil_img = Image.open(filepath)
            pil_img.thumbnail((thumb_w, thumb_h))
            tk_img = ImageTk.PhotoImage(pil_img)
            viewer.config(image=tk_img)
            viewer.image = tk_img
            name_var.set(filename)
            counter_var.set(f"{current_idx[0] + 1} / {len(images)}")
        except Exception as e:
            print(f"Gallery load error: {e}")

    def prev_image():
        current_idx[0] = (current_idx[0] - 1 + len(images)) % len(images)
        show_image()

    def next_image():
        current_idx[0] = (current_idx[0] + 1) % len(images)
        show_image()

    gal_win.bind("<Left>", lambda e: prev_image())
    gal_win.bind("<Right>", lambda e: next_image())

    btn_pad = int(SCREEN_W * 0.025)
    tk.Button(nav_frame, text="<< PREV", command=prev_image,
              font=("Arial", f_med, "bold"), bg="white",
              borderwidth=0).pack(side="left", padx=btn_pad)
    tk.Button(nav_frame, text="NEXT >>", command=next_image,
              font=("Arial", f_med, "bold"), bg="white",
              borderwidth=0).pack(side="left", padx=btn_pad)

    show_image()

def run_model_clicked():
    global last_frame, is_paused
    if is_paused and last_frame is not None:
        annotated_frame, tally = run_yolo_inference(last_frame)
        
        # Keep results on the preview screen
        display_on_label(annotated_frame)
        
        # Update Tally Box
        output = "RESULTS TALLY\n" + "="*15 + "\n"
        for label, count in tally.items():
            output += f"{label.upper()}: {count}\n"
        output += "="*15 + f"\nTOTAL: {sum(tally.values())}"
        results_label.config(text=output)
        
        # Save detection
        if not os.path.exists("processed_images"): os.makedirs("processed_images")
        cv2.imwrite(f"processed_images/det_{datetime.now().strftime('%H%M%S')}.jpg", annotated_frame)


def run_slicer_clicked():
    """Runs Roboflow Inference Slicer on the captured/uploaded frame."""
    global last_frame, is_paused
    if is_paused and last_frame is not None:
        annotated_frame, tally = run_roboflow_slicer_inference(last_frame)

        display_on_label(annotated_frame)

        output = "SLICER TALLY\n" + "="*15 + "\n"
        for label, count in tally.items():
            output += f"{label.upper()}: {count}\n"
        output += "="*15 + f"\nTOTAL: {sum(tally.values())}"
        results_label.config(text=output)

        if not os.path.exists("processed_images"): os.makedirs("processed_images")
        cv2.imwrite(f"processed_images/slicer_{datetime.now().strftime('%H%M%S')}.jpg", annotated_frame)

# =============================================================================
#  MAIN INTERFACE
# =============================================================================
def start_app():
    global camera_label, results_label, is_paused
    global SCREEN_W, SCREEN_H, PREVIEW_W, PREVIEW_H

    root = tk.Tk()
    root.title("BerryScan - InferenceSlicer")
    root.attributes("-fullscreen", True)
    root.configure(bg="#FFD782")

    # --- Get the ACTUAL screen dimensions and build scale helpers ---
    root.update_idletasks()
    SCREEN_W = root.winfo_screenwidth()
    SCREEN_H = root.winfo_screenheight()

    # Independent x/y scale so the layout fills the whole display
    sx_ratio = SCREEN_W / REF_W
    sy_ratio = SCREEN_H / REF_H
    # Uniform scale for fonts so text stays proportional and never overflows
    f_ratio = min(sx_ratio, sy_ratio)

    def SX(v): return int(round(v * sx_ratio))
    def SY(v): return int(round(v * sy_ratio))
    def F(pt): return max(8, int(round(pt * f_ratio)))

    # --- Exit bindings ---
    root.bind("<Escape>", lambda e: [cap.release(), root.destroy()])
    root.protocol("WM_DELETE_WINDOW", lambda: [cap.release(), root.destroy()])

    title_font = tkfont.Font(family="Arial", size=F(46), weight="bold")
    btn_font   = tkfont.Font(family="Arial", size=F(12), weight="bold")

    # 1. Gallery / Logo Button (top-left)
    logo_w, logo_h = SX(96), SY(96)
    try:
        pil_icon = Image.open("Button.jpg").resize((logo_w, logo_h), Image.LANCZOS)
        icon_image = ImageTk.PhotoImage(pil_icon)
        logo_btn = tk.Button(root, image=icon_image, bg="#FFD782",
                             command=gallery_button_clicked,
                             borderwidth=0, cursor="hand2")
        logo_btn.image = icon_image
    except Exception:
        logo_btn = tk.Button(root, text="GALLERY", bg="#262626", fg="white",
                             font=btn_font, command=gallery_button_clicked,
                             borderwidth=0, cursor="hand2")
    logo_btn.place(x=SX(40), y=SY(40), width=logo_w, height=logo_h)

    # 2. Branding
    tk.Label(root, text="Berry", bg="#FFD782", fg="#E82C2A",
             font=title_font).place(x=SX(40), y=SY(140))
    tk.Label(root, text="Scan", bg="#FFD782", fg="black",
             font=title_font).place(x=SX(225), y=SY(140))

    # 3. Control Buttons
    def toggle_capture():
        global is_paused
        is_paused = not is_paused
        if not is_paused:
            update_camera_feed()

    def upload_action():
        global is_paused, last_frame
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            is_paused = True
            last_frame = cv2.imread(path)
            display_on_label(last_frame)

    btn_x, btn_w, btn_h = SX(40), SX(340), SY(55)

    tk.Button(root, text="CAPTURE PHOTO", bg="#262626", fg="white",
              font=btn_font, command=toggle_capture,
              borderwidth=0).place(x=btn_x, y=SY(250), width=btn_w, height=btn_h)

    tk.Button(root, text="RUN MODEL (YOLO)", bg="#C91B1A", fg="white",
              font=btn_font, command=run_model_clicked,
              borderwidth=0).place(x=btn_x, y=SY(320), width=btn_w, height=btn_h)

    tk.Button(root, text="UPLOAD IMAGE", bg="#005A9C", fg="white",
              font=btn_font, command=upload_action,
              borderwidth=0).place(x=btn_x, y=SY(390), width=btn_w, height=btn_h)

    tk.Button(root, text="RUN MODEL (SLICER)", bg="#2E7D32", fg="white",
              font=btn_font, command=run_slicer_clicked,
              borderwidth=0).place(x=btn_x, y=SY(460), width=btn_w, height=btn_h)

    # 4. Preview Screen — sized off the actual screen so the camera fills it
    #    Height is reduced slightly to leave room for the color legend strip.
    PREVIEW_W = SX(810)
    PREVIEW_H = SY(595)
    camera_label = tk.Label(root, bg="black")
    camera_label.place(x=SX(430), y=SY(40), width=PREVIEW_W, height=PREVIEW_H)

    # 4b. Color Legend (map-style legend strip below the preview)
    legend_h = SY(40)
    legend_y = SY(40) + PREVIEW_H + SY(5)
    legend_frame = tk.Frame(root, bg="white",
                            highlightbackground="black", highlightthickness=2)
    legend_frame.place(x=SX(430), y=legend_y,
                       width=PREVIEW_W, height=legend_h)

    legend_font = tkfont.Font(family="Arial", size=F(11), weight="bold")
    swatch_size = SY(18)

    def _add_legend_item(color_hex, text):
        item = tk.Frame(legend_frame, bg="white")
        swatch = tk.Frame(item, bg=color_hex,
                          highlightbackground="black", highlightthickness=1)
        swatch.configure(width=swatch_size, height=swatch_size)
        swatch.pack_propagate(False)
        swatch.pack(side="left", padx=(SX(14), SX(8)), pady=SY(8))
        tk.Label(item, text=text, bg="white", fg="black",
                 font=legend_font).pack(side="left", padx=(0, SX(20)))
        item.pack(side="left", fill="y")

    # Hex values match the BGR colors drawn on the bounding boxes.
    _add_legend_item("#FF0000", "INFESTED")       # Red
    _add_legend_item("#00FF00", "NON-INFESTED")   # Green
    _add_legend_item("#FFFF00", "UNKNOWN")        # Yellow

    # 5. Tally Box — bottom edge aligns with the legend (SY(680)),
    #    leaving the same ~SY(40) margin from the screen bottom as the right column
    tally_frame = tk.Frame(root, bg="white",
                           highlightbackground="black", highlightthickness=2)
    tally_frame.place(x=SX(40), y=SY(530),
                      width=SX(340), height=SY(150))

    results_label = tk.Label(
        tally_frame,
        text="Ready",
        bg="white",
        font=("Courier", F(15), "bold"),
        justify="left",
        anchor="nw",
        padx=SX(15),
        pady=SY(12)
    )
    results_label.pack(fill="both", expand=True)

    # 6. Start live feed
    update_camera_feed()
    root.mainloop()


if __name__ == "__main__":
    start_app()
