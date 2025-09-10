import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
from pathlib import Path
import shutil
from ultralytics.models.sam import SAM
from datetime import datetime
import torch

class ImageAnnotationTool:
    """
    A GUI tool for creating instance segmentation annotations.

    Features:
    - Mask and Bounding Box annotation modes.
    - SAM2 integration ('Magic Click') for semi-automated segmentation.
    - Support for multiple instances per image, saved as separate mask files.
    - Zooming and panning for precise annotations.
    - Export annotations to PNG masks, YOLO segmentation format, and/or COCO JSON format.
    """
    # --- Constants ---
    SAM_MODELS = ['sam2_b.pt', 'sam2_t.pt', 'mobile_sam.pt']
    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool for Instance Segmentation")
        self.root.geometry("1200x800")

        # --- State Variables ---
        self.input_dir = ""
        self.output_dir = ""
        self.mask_dir = ""
        self.image_files = []
        self.current_index = 0
        self.instance_count = 0

        # --- Image & Canvas Variables ---
        self.current_image = None
        self.current_photo = None
        self.mask = None
        self.bboxes = []
        self.zoom_level = 1.0
        self.offset_x, self.offset_y = 0, 0
        self.panning = False
        self.pan_start_x, self.pan_start_y = 0, 0

        # --- Annotation Tool Variables ---
        self.annotation_mode = "mask"  # 'mask' or 'bbox'
        self.mask_tool_mode = "brush"  # 'brush' or 'magic_click'
        self.tool_polarity = "positive" # 'positive' or 'negative'
        self.brush_size = 10
        self.drawing = False
        self.drawn_segments = []
        self.bbox_start_coords = None
        self.current_bbox_rect = None

        # --- Export Format Variables ---
        self.export_png_var = tk.BooleanVar(value=True)
        self.export_yolo_var = tk.BooleanVar(value=True)
        self.export_coco_var = tk.BooleanVar(value=False)

        # --- UI Variables ---
        self.status_var = tk.StringVar(value="Select directories to start.")
        self.mask_status_var = tk.StringVar(value="No active mask.")
        self.annotation_mode_var = tk.StringVar(value=self.annotation_mode)
        self.tool_var = tk.StringVar(value=self.mask_tool_mode)
        self.tool_polarity_var = tk.StringVar(value=self.tool_polarity)
        self.brush_var = tk.IntVar(value=self.brush_size)
        self.model_var = tk.StringVar(value=self.SAM_MODELS[0])

        # --- AI Model ---
        self.sam_model = None

        # --- COCO Dataset Variables ---
        self.coco_data = None
        self.image_id_counter = 1
        self.annotation_id_counter = 1

        self._setup_ui()
        self._bind_events()
        self._load_sam_model(self.model_var.get())

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def _setup_ui(self):
        """Initializes and packs all the UI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Control Panels ---
        self._setup_control_panels(main_frame)

        # --- Canvas for Image Display ---
        self._setup_canvas(main_frame)
        
        # --- Instructions Label ---
        instructions = ttk.Label(main_frame, text="Controls: ←/→: Navigate | Space: Save & Next | 'n': Save Instance | Mouse Wheel: Zoom | Right-Click+Drag: Pan")
        instructions.pack(pady=(10, 0))

    def _setup_control_panels(self, parent):
        """Sets up the top control panels for directories, modes, and tools."""
        # Panel 1: Directories, Mode, SAM Model, Zoom
        control_frame1 = ttk.Frame(parent)
        control_frame1.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_frame1, text="Input Dir", command=self._select_input_dir).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame1, text="Output Dir", command=self._select_output_dir).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame1, text="Load Mask Dir", command=self._select_mask_dir).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Label(control_frame1, text="Mode:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Radiobutton(control_frame1, text="Mask", variable=self.annotation_mode_var, value="mask", command=self._change_annotation_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame1, text="Bbox", variable=self.annotation_mode_var, value="bbox", command=self._change_annotation_mode).pack(side=tk.LEFT)

        ttk.Label(control_frame1, text="Zoom:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Button(control_frame1, text="+", command=self._zoom_in, width=3).pack(side=tk.LEFT)
        ttk.Button(control_frame1, text="-", command=self._zoom_out, width=3).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(control_frame1, text="Fit", command=self.fit_to_window, width=4).pack(side=tk.LEFT, padx=(2, 0))

        # Panel 2: Tool-specific controls (Mask/Bbox)
        self.tool_control_frame = ttk.Frame(parent)
        self.tool_control_frame.pack(fill=tk.X, pady=(0, 10))
        self._setup_mask_controls(self.tool_control_frame)
        self._setup_bbox_controls(self.tool_control_frame)
        self.mask_controls_frame.pack(fill=tk.X) # Show mask controls by default

        # Panel 3: Status Labels
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_frame, textvariable=self.mask_status_var).pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.RIGHT)

    def _setup_mask_controls(self, parent):
        """Sets up the widgets specific to the 'Mask' annotation mode."""
        self.mask_controls_frame = ttk.Frame(parent)
        
        # First row: Tools and settings
        tools_frame = ttk.Frame(self.mask_controls_frame)
        tools_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(tools_frame, text="Tool:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(tools_frame, text="Brush", variable=self.tool_var, value="brush", command=self._change_tool).pack(side=tk.LEFT)
        ttk.Radiobutton(tools_frame, text="Magic Click", variable=self.tool_var, value="magic_click", command=self._change_tool).pack(side=tk.LEFT)

        ttk.Label(tools_frame, text="Polarity:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Radiobutton(tools_frame, text="Positive", variable=self.tool_polarity_var, value="positive", command=self._change_tool_polarity).pack(side=tk.LEFT)
        ttk.Radiobutton(tools_frame, text="Negative", variable=self.tool_polarity_var, value="negative", command=self._change_tool_polarity).pack(side=tk.LEFT)

        ttk.Label(tools_frame, text="Brush Size:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Scale(tools_frame, from_=1, to=50, orient=tk.HORIZONTAL, variable=self.brush_var, command=self._change_brush_size).pack(side=tk.LEFT)

        ttk.Label(tools_frame, text="SAM Model:").pack(side=tk.LEFT, padx=(20, 5))
        model_selector = ttk.Combobox(tools_frame, textvariable=self.model_var, values=self.SAM_MODELS, state="readonly", width=12)
        model_selector.pack(side=tk.LEFT, padx=(0, 5))
        model_selector.bind("<<ComboboxSelected>>", self._on_model_selected)

        # Second row: Export format checkboxes and action buttons
        export_frame = ttk.Frame(self.mask_controls_frame)
        export_frame.pack(fill=tk.X)
        
        # Export format checkboxes
        ttk.Label(export_frame, text="Export as:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(export_frame, text="PNG Mask", variable=self.export_png_var, command=self._validate_export_selection).pack(side=tk.LEFT)
        ttk.Checkbutton(export_frame, text="YOLO TXT", variable=self.export_yolo_var, command=self._validate_export_selection).pack(side=tk.LEFT)
        ttk.Checkbutton(export_frame, text="COCO JSON", variable=self.export_coco_var, command=self._validate_export_selection).pack(side=tk.LEFT)

        # Action buttons
        ttk.Button(export_frame, text="Clear Mask", command=self.clear_mask).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(export_frame, text="Save Instance & New", command=self._save_instance_and_add_new).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(export_frame, text="Load Existing Mask", command=self._load_mask_from_file).pack(side=tk.RIGHT)

    def _setup_bbox_controls(self, parent):
        """Sets up the widgets specific to the 'Bbox' annotation mode."""
        self.bbox_controls_frame = ttk.Frame(parent)
        ttk.Button(self.bbox_controls_frame, text="Clear All Bboxes", command=self.clear_bboxes).pack(side=tk.LEFT)

    def _setup_canvas(self, parent):
        """Sets up the canvas with scrollbars for displaying the image."""
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="gray")
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll.grid(row=0, column=1, sticky="ns")

    def _bind_events(self):
        """Binds all keyboard and mouse events."""
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.focus_set()

        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Button-3>", self._start_panning)
        self.canvas.bind("<B3-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-3>", self._stop_panning)
        self.canvas.bind("<MouseWheel>", self._zoom)
        self.canvas.bind("<Button-4>", self._zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self._zoom)  # Linux scroll down

    # -------------------------------------------------------------------------
    # Export Format Validation
    # -------------------------------------------------------------------------

    def _validate_export_selection(self):
        """Ensures at least one export format is selected."""
        if not (self.export_png_var.get() or self.export_yolo_var.get() or self.export_coco_var.get()):
            # If user unchecked all, automatically check PNG
            self.export_png_var.set(True)
            messagebox.showwarning("Export Format", "At least one export format must be selected. PNG format has been automatically selected.")

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _on_key_press(self, event):
        """Handles global key press events."""
        if event.keysym == "Left": self.previous_image()
        elif event.keysym == "Right": self.next_image()
        elif event.keysym == "space": self._save_and_next()
        elif event.keysym == "n": self._save_instance_and_add_new()
    
    def _on_model_selected(self, event):
        """Handles SAM model selection from the dropdown."""
        selected_model = self.model_var.get()
        self._load_sam_model(selected_model)

    def _on_canvas_press(self, event):
        """Handles the start of an action on the canvas (drawing or magic click)."""
        if self.annotation_mode == 'mask':
            if self.mask_tool_mode == "magic_click":
                self._magic_click(event)
            else:
                self._start_mask_drawing(event)
        elif self.annotation_mode == 'bbox':
            self._start_bbox_drawing(event)

    def _on_canvas_drag(self, event):
        """Handles mouse drag events for drawing."""
        if self.annotation_mode == 'mask': self._draw_mask(event)
        elif self.annotation_mode == 'bbox': self._draw_bbox(event)

    def _on_canvas_release(self, event):
        """Handles the end of a drawing action."""
        if self.annotation_mode == 'mask': self._stop_mask_drawing(event)
        elif self.annotation_mode == 'bbox': self._stop_bbox_drawing(event)

    def _change_annotation_mode(self):
        """Switches the UI and behavior between 'Mask' and 'Bbox' modes."""
        self.annotation_mode = self.annotation_mode_var.get()
        self.mask_controls_frame.pack_forget()
        self.bbox_controls_frame.pack_forget()
        if self.annotation_mode == 'mask':
            self.mask_controls_frame.pack(fill=tk.X)
        elif self.annotation_mode == 'bbox':
            self.bbox_controls_frame.pack(fill=tk.X)
        self.update_display()

    def _change_tool(self):
        self.mask_tool_mode = self.tool_var.get()
    
    def _change_tool_polarity(self):
        self.tool_polarity = self.tool_polarity_var.get()

    def _change_brush_size(self, value):
        self.brush_size = int(float(value))

    # -------------------------------------------------------------------------
    # Directory and File Handling
    # -------------------------------------------------------------------------
    
    def _select_input_dir(self):
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir = Path(directory)
            self._load_image_list()

    def _select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = Path(directory)
            (self.output_dir / "images").mkdir(exist_ok=True)
            (self.output_dir / "masks").mkdir(exist_ok=True)
            (self.output_dir / "annotations").mkdir(exist_ok=True)
            self._initialize_coco_dataset()

    def _select_mask_dir(self):
        directory = filedialog.askdirectory(title="Select Existing Mask Directory")
        if directory:
            self.mask_dir = Path(directory)
            if self.image_files:
                self.load_current_image()

    def _load_image_list(self):
        """Scans the input directory for supported image files."""
        if not self.input_dir: return
        self.image_files = sorted([
            p for p in self.input_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ])
        self.current_index = 0
        if self.image_files:
            self.load_current_image()
        else:
            messagebox.showinfo("No Images", "No supported images found in the selected directory.")
            self.status_var.set("No images found.")

    def load_current_image(self):
        """Loads the image at the current index and prepares it for annotation."""
        if not self.image_files: return
        
        image_path = self.image_files[self.current_index]
        try:
            self.current_image = cv2.imread(str(image_path))
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {image_path}\n{e}")
            return
        
        # Reset annotations for the new image
        self.mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        self.bboxes = []
        self.instance_count = 0

        # Clear the YOLO annotation file if it exists, to start fresh for this image
        if self.output_dir:
            base_name = image_path.stem
            if self.export_yolo_var.get():
                yolo_path = self.output_dir / "annotations" / f"{base_name}.txt"
                if yolo_path.exists():
                    yolo_path.unlink() # Delete the file
            # Copy original image to output directory once
            output_image_path = self.output_dir / "images" / f"{base_name}.png"
            cv2.imwrite(str(output_image_path), cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))

        if self.mask_dir: self._try_load_existing_mask()
        
        self.fit_to_window()
        self._update_status()
        self._update_mask_status()

    # -------------------------------------------------------------------------
    # COCO Dataset Initialization
    # -------------------------------------------------------------------------

    def _initialize_coco_dataset(self):
        """Initializes the COCO dataset structure."""
        self.coco_data = {
            "info": {
                "description": "Instance segmentation annotations",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Image Annotation Tool",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "object",
                    "supercategory": "thing"
                }
            ]
        }
        self.image_id_counter = 1
        self.annotation_id_counter = 1

    # -------------------------------------------------------------------------
    # Annotation Saving & Navigation
    # -------------------------------------------------------------------------

    def _save_instance_and_add_new(self):
        """Saves the current mask as an instance and clears the canvas for a new one."""
        if self.annotation_mode != 'mask':
            messagebox.showwarning("Warning", "This feature is only available in Mask mode.")
            return

        if self._save_current_mask_instance():
            self.instance_count += 1
            self.clear_mask()
            self.status_var.set(f"Saved instance {self.instance_count - 1}. Ready for next.")

    def _save_and_next(self):
        """Saves the final annotation for the current image and moves to the next."""
        if not self.output_dir or self.current_image is None:
            messagebox.showerror("Error", "Please select an output directory and load an image.")
            return

        if self.annotation_mode == 'mask':
            # Save even if the mask is empty, to signal completion and move on
            self._save_current_mask_instance()
            # Save COCO annotations for the current image if enabled
            if self.export_coco_var.get():
                self._save_coco_annotations()
        elif self.annotation_mode == 'bbox':
            if not self.bboxes:
                messagebox.showwarning("Warning", "No bounding boxes drawn. Nothing to save.")
                return
            self._save_bbox_annotation()
        
        self.next_image()

    def _save_current_mask_instance(self):
        """Core logic to save the current mask and append its annotations."""
        if np.sum(self.mask) == 0:
            return False # Nothing to save
        if not self.output_dir:
            return False

        base_name = self.image_files[self.current_index].stem
        h, w = self.current_image.shape[:2]

        # Save PNG mask if enabled
        if self.export_png_var.get():
            mask_path = self.output_dir / "masks" / f"{base_name}_{self.instance_count}.png"
            cv2.imwrite(str(mask_path), self.mask)

        # Save YOLO format if enabled
        if self.export_yolo_var.get():
            self._save_yolo_annotation(base_name, h, w)

        # COCO format is handled separately when moving to next image
        return True

    def _save_yolo_annotation(self, base_name, h, w):
        """Saves the current mask in YOLO segmentation format."""
        # Find, process, and append contours to the main YOLO txt file
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yolo_path = self.output_dir / "annotations" / f"{base_name}.txt"
        
        yolo_strings = []
        for contour in contours:
            if cv2.contourArea(contour) < 10: continue
            
            # Normalize points and create YOLO string
            normalized_points = contour.flatten().astype(float)
            normalized_points[0::2] /= w
            normalized_points[1::2] /= h
            points_str = " ".join(map(str, normalized_points))
            yolo_strings.append(f"0 {points_str}")

        if yolo_strings:
            with open(yolo_path, 'a') as f:
                f.write("\n".join(yolo_strings) + "\n")

    def _save_coco_annotations(self):
        """Saves COCO format annotations for the current image."""
        if not self.export_coco_var.get():
            return

        base_name = self.image_files[self.current_index].stem
        h, w = self.current_image.shape[:2]

        # Add image info to COCO dataset
        image_info = {
            "id": self.image_id_counter,
            "width": w,
            "height": h,
            "file_name": f"{base_name}.png",
            "license": 1,
            "date_captured": datetime.now().isoformat()
        }
        self.coco_data["images"].append(image_info)

        # Find all mask files for this image and add annotations
        mask_dir = self.output_dir / "masks"
        if mask_dir.exists():
            for mask_file in mask_dir.glob(f"{base_name}_*.png"):
                mask_image = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask_image is not None:
                    self._add_coco_annotation(mask_image, self.image_id_counter)

        self.image_id_counter += 1

        # Save COCO JSON file
        coco_path = self.output_dir / "annotations" / "annotations.json"
        with open(coco_path, 'w') as f:
            json.dump(self.coco_data, f, indent=2)

    def _add_coco_annotation(self, mask, image_id):
        """Adds a single mask annotation to the COCO dataset."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue

            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Convert contour to segmentation format
            segmentation = contour.flatten().tolist()
            
            annotation = {
                "id": self.annotation_id_counter,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [segmentation],
                "area": area,
                "bbox": [x, y, w, h],
                "iscrowd": 0
            }
            
            self.coco_data["annotations"].append(annotation)
            self.annotation_id_counter += 1

    def _save_bbox_annotation(self):
        """Saves bounding box data to a YOLO format .txt file."""
        base_name = self.image_files[self.current_index].stem
        h, w = self.current_image.shape[:2]
        yolo_path = self.output_dir / "annotations" / f"{base_name}.txt"
        
        yolo_annotations = []
        for (x1, y1, x2, y2) in self.bboxes:
            box_w, box_h = x2 - x1, y2 - y1
            center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
            norm_cx, norm_cy = center_x / w, center_y / h
            norm_w, norm_h = box_w / w, box_h / h
            yolo_annotations.append(f"0 {norm_cx} {norm_cy} {norm_w} {norm_h}")
        
        with open(yolo_path, 'w') as f:
            f.write("\n".join(yolo_annotations))

    def previous_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
        elif self.image_files:
            messagebox.showinfo("Done", "You have annotated the last image!")

    # -------------------------------------------------------------------------
    # Drawing & Annotation Logic
    # -------------------------------------------------------------------------

    def _load_sam_model(self, model_name):
        """Loads the specified SAM model."""
        try:
            self.status_var.set(f"Loading SAM model: {model_name}...")
            self.root.update_idletasks()
            self.sam_model = SAM(model_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.sam_model.to(device)
            self.status_var.set(f"SAM model '{model_name}' loaded successfully on {device}.")
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load SAM model: {model_name}\n\n{e}\n\nPlease ensure the model file is in the correct directory.")
            self.status_var.set(f"Failed to load model: {model_name}")
    
    def _magic_click(self, event):
        """Performs segmentation using the SAM model based on a user click."""
        if not self.sam_model:
            messagebox.showwarning("Warning", "SAM model is not loaded.")
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        img_x, img_y = self._canvas_to_image_coords(canvas_x, canvas_y)

        h, w = self.current_image.shape[:2]
        if not (0 <= img_x < w and 0 <= img_y < h): return

        self.status_var.set("Running SAM inference...")
        self.root.update_idletasks()
        
        results = self.sam_model.predict(self.current_image, points=[[img_x, img_y]], labels=[1])
        
        if results[0].masks is not None:
            new_mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
            if self.tool_polarity == "positive":
                self.mask = cv2.bitwise_or(self.mask, new_mask)
            else: # Negative
                self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(new_mask))
            self.update_display()
            self._update_mask_status()
        
        self._update_status()

    # --- Mask Drawing ---
    def _start_mask_drawing(self, event):
        self.drawing = True
        # --- FIX: Use canvas coordinates ---
        self.last_x = self.canvas.canvasx(event.x)
        self.last_y = self.canvas.canvasy(event.y)
        # --- End of FIX ---
        self.drawn_segments = []
        
    def _draw_mask(self, event):
        if not self.drawing: return

        # --- FIX: Get current canvas coordinates ---
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        # --- End of FIX ---

        fill_color = "green" if self.tool_polarity == "positive" else "red"
        # --- FIX: Use canvas coordinates for drawing ---
        self.canvas.create_line(self.last_x, self.last_y, current_x, current_y, 
                                fill=fill_color, width=self.brush_size, capstyle=tk.ROUND, 
                                smooth=True, tags="drawing_line")
        self.drawn_segments.append((self.last_x, self.last_y, current_x, current_y))
        self.last_x, self.last_y = current_x, current_y
        # --- End of FIX ---
        
    def _stop_mask_drawing(self, event):
        # This method is already correct because it processes self.drawn_segments,
        # which will now be populated with the correct canvas coordinates from _draw_mask.
        # No changes needed here.
        if not self.drawing: return
        self.drawing = False
        if self.drawn_segments:
            mask_value = 255 if self.tool_polarity == "positive" else 0
            draw_brush_size = max(1, int(self.brush_size / self.zoom_level))
            for x1, y1, x2, y2 in self.drawn_segments:
                last_img_x, last_img_y = self._canvas_to_image_coords(x1, y1)
                img_x, img_y = self._canvas_to_image_coords(x2, y2)
                cv2.line(self.mask, (last_img_x, last_img_y), (img_x, img_y), 
                         mask_value, draw_brush_size)
        self.canvas.delete("drawing_line")
        self.update_display()
        self._update_mask_status()

    # --- Bbox Drawing ---
    def _start_bbox_drawing(self, event):
        self.drawing = True
        # --- FIX: Use canvas coordinates ---
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.bbox_start_coords = (canvas_x, canvas_y)
        # --- End of FIX ---
        self.current_bbox_rect = self.canvas.create_rectangle(
            *self.bbox_start_coords, *self.bbox_start_coords, 
            outline="red", width=2, tags="bbox_rect")

    def _draw_bbox(self, event):
        if not self.drawing: return
        # --- FIX: Use canvas coordinates ---
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.current_bbox_rect, *self.bbox_start_coords, current_x, current_y)
        # --- End of FIX ---
        
    def _stop_bbox_drawing(self, event):
        # This method is also already correct because it reads final coordinates
        # from the canvas object itself, which are already in canvas space.
        # No changes needed here.
        if not self.drawing: return
        self.drawing = False
        x1, y1, x2, y2 = self.canvas.coords(self.current_bbox_rect)
        self.canvas.delete(self.current_bbox_rect)
        self.current_bbox_rect = None

        img_x1, img_y1 = self._canvas_to_image_coords(x1, y1)
        img_x2, img_y2 = self._canvas_to_image_coords(x2, y2)
        
        final_x1, final_y1 = min(img_x1, img_x2), min(img_y1, img_y2)
        final_x2, final_y2 = max(img_x1, img_x2), max(img_y1, img_y2)

        if final_x2 - final_x1 > 1 and final_y2 - final_y1 > 1:
            self.bboxes.append((final_x1, final_y1, final_x2, final_y2))
        
        self.update_display()

    # -------------------------------------------------------------------------
    # Canvas View Controls (Zoom, Pan, Display)
    # -------------------------------------------------------------------------
    def update_display(self):
        """Redraws the canvas with the current image, mask, and annotations."""
        if self.current_image is None: return

        h, w = self.current_image.shape[:2]
        display_w = int(w * self.zoom_level)
        display_h = int(h * self.zoom_level)

        if self.annotation_mode == 'mask' and self.mask is not None and np.sum(self.mask) > 0:
            colored_mask = np.zeros_like(self.current_image)
            colored_mask[self.mask > 0] = [0, 255, 0] # Green overlay
            blended = cv2.addWeighted(self.current_image, 0.7, colored_mask, 0.3, 0)
            display_image_data = blended
        else:
            display_image_data = self.current_image

        resized_image = cv2.resize(display_image_data, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
        pil_image = Image.fromarray(resized_image)
        self.current_photo = ImageTk.PhotoImage(pil_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.current_photo)
        
        if self.annotation_mode == 'bbox':
            for x1, y1, x2, y2 in self.bboxes:
                c_x1, c_y1 = self._image_to_canvas_coords(x1, y1)
                c_x2, c_y2 = self._image_to_canvas_coords(x2, y2)
                self.canvas.create_rectangle(c_x1, c_y1, c_x2, c_y2, outline="cyan", width=2)
        
        self.canvas.configure(scrollregion=(self.offset_x, self.offset_y, self.offset_x + display_w, self.offset_y + display_h))
        self._update_status()

    def fit_to_window(self):
        if self.current_image is None: return
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <=1: canvas_w, canvas_h = 800, 600 # Default if not drawn yet

        h, w = self.current_image.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        self.zoom_level = scale
        self.offset_x, self.offset_y = 0, 0
        self.update_display()

    def _zoom_in(self): self._zoom(delta=120)
    def _zoom_out(self): self._zoom(delta=-120)

    def _zoom(self, event=None, delta=None):
        if self.current_image is None: return
        if delta is None: delta = event.delta

        mouse_x = self.canvas.canvasx(event.x) if event else self.canvas.winfo_width() / 2
        mouse_y = self.canvas.canvasy(event.y) if event else self.canvas.winfo_height() / 2
        
        zoom_factor = 1.1 if delta > 0 else 0.9
        old_zoom = self.zoom_level
        self.zoom_level = max(0.1, min(5.0, self.zoom_level * zoom_factor))

        if old_zoom != self.zoom_level:
            zoom_ratio = self.zoom_level / old_zoom
            self.offset_x = mouse_x - (mouse_x - self.offset_x) * zoom_ratio
            self.offset_y = mouse_y - (mouse_y - self.offset_y) * zoom_ratio
            self.update_display()

    def _start_panning(self, event):
        self.panning, self.pan_start_x, self.pan_start_y = True, event.x, event.y
        
    def _pan(self, event):
        if not self.panning: return
        dx, dy = event.x - self.pan_start_x, event.y - self.pan_start_y
        self.offset_x += dx
        self.offset_y += dy
        self.pan_start_x, self.pan_start_y = event.x, event.y
        self.update_display()

    def _stop_panning(self, event): self.panning = False

    # -------------------------------------------------------------------------
    # Utility & Helper Methods
    # -------------------------------------------------------------------------
    def clear_mask(self):
        if self.mask is not None:
            self.mask.fill(0)
            self.update_display()
            self._update_mask_status()

    def clear_bboxes(self):
        self.bboxes = []
        self.update_display()

    def _update_status(self):
        if self.image_files:
            filename = self.image_files[self.current_index].name
            zoom_percent = int(self.zoom_level * 100)
            export_formats = []
            if self.export_png_var.get(): export_formats.append("PNG")
            if self.export_yolo_var.get(): export_formats.append("YOLO")
            if self.export_coco_var.get(): export_formats.append("COCO")
            formats_str = "+".join(export_formats)
            self.status_var.set(f"Image {self.current_index + 1}/{len(self.image_files)}: {filename} (Zoom: {zoom_percent}%) Export: {formats_str}")

    def _update_mask_status(self):
        if self.mask is not None and np.any(self.mask):
            annotated_pixels = np.sum(self.mask > 0)
            percentage = (annotated_pixels / self.mask.size) * 100
            self.mask_status_var.set(f"Mask active ({percentage:.2f}% annotated). Instance: {self.instance_count}")
        else:
            self.mask_status_var.set(f"No active mask. Ready for instance: {self.instance_count}")

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        return int((canvas_x - self.offset_x) / self.zoom_level), int((canvas_y - self.offset_y) / self.zoom_level)
    
    def _image_to_canvas_coords(self, img_x, img_y):
        return img_x * self.zoom_level + self.offset_x, img_y * self.zoom_level + self.offset_y
    
    def _load_mask_from_file(self):
        """Command for the 'Load Mask' button. Tries to load a mask and gives user feedback."""
        if not self.mask_dir:
            messagebox.showwarning("Warning", "Please select a mask directory first.")
            return

        if self._try_load_existing_mask():
            messagebox.showinfo("Success", "Mask loaded successfully.")
            self.update_display()
            self._update_mask_status()
        else:
            image_name = self.image_files[self.current_index].name
            messagebox.showerror("Error", f"Could not find a mask for '{image_name}' in '{self.mask_dir}'.")

    def _try_load_existing_mask(self):
        """Attempts to find and load a mask corresponding to the current image."""
        if not self.mask_dir or not self.image_files: return False

        image_path = self.image_files[self.current_index]
        possible_mask_names = [f"predicted_{image_path.stem}.png", f"{image_path.stem}.png"]
        mask_path = None
        for name in possible_mask_names:
            path_to_check = self.mask_dir / name
            if path_to_check.exists():
                mask_path = path_to_check
                break
        
        if not mask_path: return False

        try:
            loaded_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if loaded_mask is None: return False

            h, w = self.current_image.shape[:2]
            if loaded_mask.shape != (h, w):
                loaded_mask = cv2.resize(loaded_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            _, self.mask = cv2.threshold(loaded_mask, 1, 255, cv2.THRESH_BINARY)
            return True
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return False

def main():
    root = tk.Tk()
    app = ImageAnnotationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()