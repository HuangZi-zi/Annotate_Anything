import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
from pathlib import Path
import re
from ultralytics.models.sam import SAM
from ultralytics import YOLO
from datetime import datetime
import torch
from scipy.spatial import distance

class ImageAnnotationTool:
    """
    A GUI tool for creating and editing polygon-based instance segmentation annotations.

    Features:
    - Click-to-Edit: Directly click on any saved polygon to load it for modification.
    - Annotations are stored as polygons with user-defined class names.
    - Interactive tools: Brush, Polygon, and 'Magic Click' (SAM integration).
    - Support for multiple instances and classes per image.
    - Zooming and panning for precise annotations.
    - Export formats: PNG masks, YOLO TXT, and COCO JSON.
    """
    # --- Constants ---
    SAM_MODELS = ['sam2_b.pt', 'sam2_t.pt', 'mobile_sam.pt']
    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    CLASS_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 128, 0), (128, 0, 255)]


    def __init__(self, root):
        self.root = root
        self.root.title("Polygon-Based Image Annotation Tool")
        self.root.geometry("1200x800")

        # --- State Variables ---
        self.input_dir = ""
        self.output_dir = ""
        self.mask_dir = ""
        self.image_files = []
        self.current_index = 0
        self.annotations = []
        self.class_names = {}
        self.editing_instance_index = None # Holds the index of the instance being edited

        # --- Image & Canvas Variables ---
        self.current_image = None
        self.current_photo = None
        self.mask = None # Temporary mask for active drawing/editing
        self.polygon_points = []
        self.polygon_mask = None
        self.zoom_level = 1.0
        self.offset_x, self.offset_y = 0, 0
        self.panning = False
        self.pan_start_x, self.pan_start_y = 0, 0

        # --- Annotation Tool Variables ---
        self.mask_tool_mode = "brush"
        self.tool_polarity = "positive"
        self.brush_size = 10
        self.drawing = False
        self.drawn_segments = []
        self.REMOVAL_DISTANCE_THRESHOLD = 5
        self.current_class = tk.IntVar(value=0)
        
        # --- Export Format Variables ---
        self.export_png_var = tk.BooleanVar(value=False)
        self.export_yolo_var = tk.BooleanVar(value=True)
        self.export_coco_var = tk.BooleanVar(value=True)

        # --- UI Variables ---
        self.status_var = tk.StringVar(value="Select directories to start.")
        self.mask_status_var = tk.StringVar(value="No active mask.")
        self.tool_var = tk.StringVar(value=self.mask_tool_mode)
        self.tool_polarity_var = tk.StringVar(value=self.tool_polarity)
        self.brush_var = tk.IntVar(value=self.brush_size)
        self.model_var = tk.StringVar(value=self.SAM_MODELS[0])
        self.class_name_var = tk.StringVar()

        # --- AI Model ---
        self.sam_model = None
        self.yolo_model = None
        self.yolo_model_dir = None

        # --- COCO Dataset Variables ---
        self.coco_data = None
        self.image_id_counter = 1
        self.annotation_id_counter = 1

        self._setup_ui()
        self._bind_events()
        # self._load_sam_model(self.model_var.get())

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def _setup_ui(self):
        """Initializes and packs all the UI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._setup_control_panels(main_frame)
        self._setup_canvas(main_frame)
        
        instructions = ttk.Label(main_frame, text="Controls: ←/→: Navigate | Space: Save & Next | 'i': Save Instance | 'p': Finalize Polygon | Mouse Wheel: Zoom | Right-Click+Drag: Pan")
        instructions.pack(pady=(10, 0))

    def _setup_control_panels(self, parent):
        """Sets up the top control panels for directories, modes, and tools."""
        control_frame1 = ttk.Frame(parent)
        control_frame1.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_frame1, text="Input Dir", command=self._select_input_dir).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame1, text="Output Dir", command=self._select_output_dir).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame1, text="Load Mask Dir", command=self._select_mask_dir).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame1, text="Load Model Dir", command=self._select_model_dir).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Label(control_frame1, text="Zoom:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Button(control_frame1, text="+", command=self._zoom_in, width=3).pack(side=tk.LEFT)
        ttk.Button(control_frame1, text="-", command=self._zoom_out, width=3).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(control_frame1, text="Fit", command=self.fit_to_window, width=4).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(control_frame1, text="Current Class:").pack(side=tk.LEFT, padx=(20, 5))
        self.class_label = ttk.Label(control_frame1, textvariable=self.current_class, width=3, anchor="center")
        self.class_label.pack(side=tk.LEFT)
        ttk.Button(control_frame1, text="+", command=self._next_class, width=3).pack(side=tk.LEFT)
        ttk.Button(control_frame1, text="-", command=self._previous_class, width=3).pack(side=tk.LEFT)
        
        self.class_name_entry = ttk.Entry(control_frame1, textvariable=self.class_name_var, width=15)
        self.class_name_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.class_name_entry.bind("<FocusOut>", self._save_current_class_name)
        self.class_name_entry.bind("<Return>", self._save_current_class_name)

        self.tool_control_frame = ttk.Frame(parent)
        self.tool_control_frame.pack(fill=tk.X, pady=(0, 10))
        self._setup_mask_controls(self.tool_control_frame)
        self.mask_controls_frame.pack(fill=tk.X)

        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_frame, textvariable=self.mask_status_var).pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.RIGHT)

    def _next_class(self):
        self._save_current_class_name()
        self.current_class.set(self.current_class.get() + 1)
        self._update_class_display()

    def _previous_class(self):
        if self.current_class.get() > 0:
            self._save_current_class_name()
            self.current_class.set(self.current_class.get() - 1)
            self._update_class_display()
    
    def _save_current_class_name(self, event=None):
        class_id = self.current_class.get()
        class_name = self.class_name_var.get().strip()
        if class_name: self.class_names[class_id] = class_name
        if event: self.root.focus_set()

    def _update_class_display(self):
        class_id = self.current_class.get()
        name = self.class_names.get(class_id, f"class_{class_id}")
        self.class_name_var.set(name)
        color = self.CLASS_COLORS[class_id % len(self.CLASS_COLORS)]
        hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        self.class_label.configure(background=hex_color)


    def _setup_mask_controls(self, parent):
        self.mask_controls_frame = ttk.Frame(parent)
        
        tools_frame = ttk.Frame(self.mask_controls_frame)
        tools_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(tools_frame, text="Tool:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(tools_frame, text="Brush", variable=self.tool_var, value="brush", command=self._change_tool).pack(side=tk.LEFT)
        ttk.Radiobutton(tools_frame, text="Polygon", variable=self.tool_var, value="polygon", command=self._change_tool).pack(side=tk.LEFT)
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

        export_frame = ttk.Frame(self.mask_controls_frame)
        export_frame.pack(fill=tk.X)
        
        ttk.Label(export_frame, text="Export as:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(export_frame, text="(Semantic)PNG Masks", variable=self.export_png_var).pack(side=tk.LEFT)
        ttk.Checkbutton(export_frame, text="YOLO TXT", variable=self.export_yolo_var).pack(side=tk.LEFT)
        ttk.Checkbutton(export_frame, text="COCO JSON", variable=self.export_coco_var).pack(side=tk.LEFT)

        ttk.Button(export_frame, text="Clear Active Mask", command=self.clear_mask).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(export_frame, text="Clear All Annotations", command=self.clear_all_annotations).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(export_frame, text="Save Instance ('i')", command=self._save_instance).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(export_frame, text="Load Existing Mask", command=self._load_mask_from_file).pack(side=tk.RIGHT)

    def _setup_canvas(self, parent):
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        canvas_frame.grid_rowconfigure(0, weight=1); canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, bg="gray")
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll.grid(row=0, column=1, sticky="ns")

    def _bind_events(self):
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.focus_set()
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Button-3>", self._start_panning)
        self.canvas.bind("<B3-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-3>", self._stop_panning)
        self.canvas.bind("<MouseWheel>", self._zoom)
        self.canvas.bind("<Button-4>", self._zoom); self.canvas.bind("<Button-5>", self._zoom)

    # -------------------------------------------------------------------------
    # Event Handlers & UI Callbacks
    # -------------------------------------------------------------------------

    def _on_key_press(self, event):
        if isinstance(self.root.focus_get(), ttk.Entry): return
        if event.keysym == "Left": self.previous_image()
        elif event.keysym == "Right": self.next_image()
        elif event.keysym == "space": self._save_and_next()
        elif event.keysym == "i": self._save_instance()
        elif event.keysym == "p": self._merge_polygon_into_mask()
    
    def _on_model_selected(self, event): self._load_sam_model(self.model_var.get())

    def _on_canvas_press(self, event):
        # First, check if the user is clicking an existing instance to edit it
        if self._select_instance_for_editing(event):
            return # If an instance was selected, don't start a new drawing

        # If not editing, proceed with the normal drawing tools
        if self.mask_tool_mode == "magic_click": self._magic_click(event)
        elif self.mask_tool_mode == "polygon": self._polygon_click(event)
        else: self._start_mask_drawing(event)

    def _on_canvas_drag(self, event):
        if self.mask_tool_mode == "brush": self._draw_mask(event)

    def _on_canvas_release(self, event):
        if self.mask_tool_mode == "brush": self._stop_mask_drawing(event)

    def _change_tool(self): self.mask_tool_mode = self.tool_var.get()
    def _change_tool_polarity(self): self.tool_polarity = self.tool_polarity_var.get()
    def _change_brush_size(self, value): self.brush_size = int(float(value))

    # -------------------------------------------------------------------------
    # Directory and File Handling
    # -------------------------------------------------------------------------
    
    def _select_input_dir(self):
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory: self.input_dir = Path(directory); self._load_image_list()

    def _select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = Path(directory)
            (self.output_dir / "images").mkdir(exist_ok=True, parents=True)
            (self.output_dir / "masks").mkdir(exist_ok=True, parents=True)
            (self.output_dir / "annotations").mkdir(exist_ok=True, parents=True)
            self._initialize_coco_dataset()

    def _select_mask_dir(self):
        directory = filedialog.askdirectory(title="Select Existing Mask Directory")
        if directory: self.mask_dir = Path(directory)

    def _select_model_dir(self):
        directory = filedialog.askopenfilename(title="Select YOLO Model File")
        if directory:
            self.yolo_model_dir = Path(directory)
            self.yolo_model = YOLO(self.yolo_model_dir)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.yolo_model.to(device)

    def _load_image_list(self):
        if not self.input_dir: return
        self.image_files = sorted([p for p in self.input_dir.rglob('*') if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS])
        if self.image_files: self.current_index = 0; self.load_current_image()
        else: messagebox.showinfo("No Images", "No supported images found in the selected directory.")

    def load_current_image(self):
        if not self.image_files: return
        image_path = self.image_files[self.current_index]
        try:
            self.current_image = cv2.imread(str(image_path))
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        except Exception as e: messagebox.showerror("Error", f"Failed to load image: {image_path}\n{e}"); return
        
        # Reset annotations and state for the new image
        self.current_class.set(0); self.class_names = {}; self._update_class_display()
        self.mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        self.polygon_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        self.annotations = []; self.polygon_points = []; self.editing_instance_index = None

        if self.output_dir:
            base_name = image_path.stem
            yolo_path = self.output_dir / "annotations" / f"{base_name}.txt"
            if yolo_path.exists(): yolo_path.unlink()
            output_image_path = self.output_dir / "images" / f"{base_name}.png"
            cv2.imwrite(str(output_image_path), cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))

        if self.mask_dir: self._try_load_existing_mask()
        if self.yolo_model: self._try_run_yolo_model()
        self.fit_to_window()

    # -------------------------------------------------------------------------
    # Annotation Saving & Navigation
    # -------------------------------------------------------------------------

    def _save_instance(self):
        """Saves the current active mask (new or edited) as a permanent polygon annotation."""
        self._save_current_class_name()
        self._merge_polygon_into_mask()

        if np.sum(self.mask) == 0:
            # messagebox.showinfo("Info", "No active mask to save as an instance.")
            self.clear_mask()
            return

        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: self.clear_mask(); return

        simplified_polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < 10: continue
            epsilon = 0.005 * cv2.arcLength(contour, True)
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            if len(simplified_contour) >= 3:
                simplified_polygons.append(simplified_contour.squeeze(axis=1).tolist())

        if not simplified_polygons: self.clear_mask(); return

        new_annotation = {"class_id": self.current_class.get(), "polygons": simplified_polygons}
        
        # If we were editing, the original instance was already removed.
        # This new one is simply added to the list.
        self.annotations.append(new_annotation)
        
        edit_str = f" (edited instance {self.editing_instance_index})" if self.editing_instance_index is not None else ""
        self.status_var.set(f"Saved instance {len(self.annotations)}{edit_str}.")
        self.editing_instance_index = None # Reset editing state
        self.clear_mask()

    def _save_and_next(self):
        if not self.output_dir or self.current_image is None: messagebox.showerror("Error", "Select output dir and load image."); return
        if np.sum(self.mask) > 0: self._save_instance()
        if not self.annotations: self.next_image(); return
            
        base_name = self.image_files[self.current_index].stem
        h, w = self.current_image.shape[:2]

        if self.export_png_var.get(): self._save_png_masks(base_name, h, w)
        if self.export_yolo_var.get(): self._save_yolo_annotation(base_name, h, w)
        if self.export_coco_var.get(): self._add_image_to_coco(base_name, h, w)
        
        self.next_image()

    def _save_png_masks(self, base_name, h, w):
        output_masks_dir = self.output_dir / "masks"
        for i, ann in enumerate(self.annotations):
            instance_mask = np.zeros((h, w), dtype=np.uint8)
            class_id = ann['class_id']+1
            cv2.fillPoly(instance_mask, [np.array(p, dtype=np.int32) for p in ann['polygons']], class_id*10)
        mask_path = output_masks_dir / f"{base_name}.png"
        cv2.imwrite(str(mask_path), instance_mask)

    def _save_yolo_annotation(self, base_name, h, w):
        yolo_path = self.output_dir / "annotations" / f"{base_name}.txt"
        yolo_strings = []
        for ann in self.annotations:
            class_id = ann['class_id']
            for poly in ann['polygons']:
                normalized_points = np.array(poly, dtype=float).flatten()
                normalized_points[0::2] /= w  # Normalize x
                normalized_points[1::2] /= h  # Normalize y
                points_str = " ".join(map(str, normalized_points))
                yolo_strings.append(f"{class_id} {points_str}")

        if yolo_strings:
            with open(yolo_path, 'w') as f:
                f.write("\n".join(yolo_strings) + "\n")

    def _add_image_to_coco(self, base_name, h, w):
        self.coco_data["images"].append({"id": self.image_id_counter, "width": w, "height": h, "file_name": f"{base_name}.png"})
        for ann in self.annotations:
            for poly in ann['polygons']:
                poly_np = np.array(poly, dtype=np.int32)
                x, y, w_bbox, h_bbox = cv2.boundingRect(poly_np)
                self.coco_data["annotations"].append({
                    "id": self.annotation_id_counter, "image_id": self.image_id_counter,
                    "category_id": ann['class_id'] + 1, "segmentation": [poly_np.flatten().tolist()],
                    "area": float(cv2.contourArea(poly_np)), "bbox": [x, y, w_bbox, h_bbox], "iscrowd": 0 })
                self.annotation_id_counter += 1
        self.image_id_counter += 1

    def _finalize_coco(self):
        if self.coco_data and self.export_coco_var.get() and self.coco_data['annotations']:
            all_class_ids = {ann['category_id'] - 1 for ann in self.coco_data['annotations']}
            self.coco_data['categories'] = [{'id': i + 1, 'name': self.class_names.get(i, f"class_{i}"), 'supercategory': 'object'} for i in sorted(list(all_class_ids))]
            coco_path = self.output_dir / "annotations" / "annotations.json"
            with open(coco_path, 'w') as f: json.dump(self.coco_data, f, indent=2)
            print(f"COCO annotations saved to {coco_path}")

    def previous_image(self):
        if self.image_files and self.current_index > 0: self.current_index -= 1; self.load_current_image()
    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1: self.current_index += 1; self.load_current_image()
        elif self.image_files: self._finalize_coco(); messagebox.showinfo("Done", "You have annotated the last image!")

    # -------------------------------------------------------------------------
    # Annotation Editing Logic
    # -------------------------------------------------------------------------

    def _get_instance_at_point(self, x, y):
        """Finds the topmost instance annotation under the given image coordinates."""
        for i in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[i]
            for poly in ann['polygons']:
                if cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (x, y), False) >= 0:
                    return i
        return None

    def _select_instance_for_editing(self, event):
        """Checks for a click on an existing instance and loads it for editing."""
        if np.any(self.mask) or self.polygon_points: return False # Don't select if there's an active drawing

        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        img_x, img_y = self._canvas_to_image_coords(canvas_x, canvas_y)

        instance_idx = self._get_instance_at_point(img_x, img_y)
        if instance_idx is not None:
            self.editing_instance_index = instance_idx
            
            # Remove instance from list and load its data
            instance_to_edit = self.annotations.pop(instance_idx)
            class_id = instance_to_edit['class_id']
            polygons = [np.array(p, dtype=np.int32) for p in instance_to_edit['polygons']]

            # Set the tool to the correct class
            self.current_class.set(class_id)
            self._update_class_display()
            
            # Load polygons into the active mask
            self.mask.fill(0)
            cv2.fillPoly(self.mask, polygons, 255)
            
            self.mask_status_var.set(f"Editing instance {instance_idx}. Press 'i' to save changes.")
            self.update_display()
            return True # Signifies that an instance was selected
            
        return False # No instance was selected

    # -------------------------------------------------------------------------
    # Drawing & Annotation Logic
    # -------------------------------------------------------------------------

    def _load_sam_model(self, model_name):
        try:
            self.status_var.set(f"Loading SAM model: {model_name}...")
            self.root.update_idletasks()
            self.sam_model = SAM(model_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.sam_model.to(device)
            self.status_var.set(f"SAM model '{model_name}' loaded successfully on {device}.")
        except Exception as e: messagebox.showerror("Model Error", f"Could not load SAM model: {model_name}\n\n{e}"); self.status_var.set(f"Failed to load model: {model_name}")
    
    def _magic_click(self, event):
        if not self.sam_model: return
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        img_x, img_y = self._canvas_to_image_coords(canvas_x, canvas_y)
        h, w = self.current_image.shape[:2]
        if not (0 <= img_x < w and 0 <= img_y < h): return

        self.status_var.set("Running SAM inference...")
        self.root.update_idletasks()
        results = self.sam_model.predict(self.current_image, points=[[img_x, img_y]], labels=[1])
        if results and results[0].masks:
            new_mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
            if self.tool_polarity == "positive": self.mask = cv2.bitwise_or(self.mask, new_mask)
            else: self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(new_mask))
            self.update_display()
        self._update_status()

    def _polygon_click(self, event):
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        img_x, img_y = self._canvas_to_image_coords(canvas_x, canvas_y)
        h, w = self.current_image.shape[:2]
        if not (0 <= img_x < w and 0 <= img_y < h): return

        clicked_point = (img_x, img_y)
        if self.polygon_points:
            distances = [distance.euclidean(clicked_point, p) for p in self.polygon_points]
            min_dist_idx = np.argmin(distances)
            if distances[min_dist_idx] < self.REMOVAL_DISTANCE_THRESHOLD / self.zoom_level: self.polygon_points.pop(min_dist_idx)
            else: self.polygon_points.append(list(clicked_point))
        else: self.polygon_points.append(list(clicked_point))
        self._update_mask_with_polygon(); self.update_display()

    def _update_mask_with_polygon(self):
        self.polygon_mask.fill(0)
        if len(self.polygon_points) >= 3:
            pts = np.array(self.polygon_points, dtype=np.int32)
            centroid = np.mean(pts, axis=0)
            angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
            pts = pts[np.argsort(angles)]
            cv2.fillPoly(self.polygon_mask, [pts], 255)
            # cv2.fillPoly(self.polygon_mask, [np.array(self.polygon_points, dtype=np.int32)], 255)

    def _merge_polygon_into_mask(self):
        if np.sum(self.polygon_mask) > 0:
            if self.tool_polarity == "positive": self.mask = cv2.bitwise_or(self.mask, self.polygon_mask)
            else: self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(self.polygon_mask))
            self.polygon_mask.fill(0); self.polygon_points = []; self.update_display()

    def _start_mask_drawing(self, event): self.drawing = True; self.last_x, self.last_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y); self.drawn_segments = []
    def _draw_mask(self, event):
        if not self.drawing: return
        current_x, current_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.create_line(self.last_x, self.last_y, current_x, current_y, fill="green" if self.tool_polarity == "positive" else "red", width=self.brush_size, capstyle=tk.ROUND, smooth=True, tags="drawing_line")
        self.drawn_segments.append((self.last_x, self.last_y, current_x, current_y)); self.last_x, self.last_y = current_x, current_y
        
    def _stop_mask_drawing(self, event):
        if not self.drawing: return
        self.drawing = False
        if self.drawn_segments:
            mask_value = 255 if self.tool_polarity == "positive" else 0
            draw_brush_size = max(1, int(self.brush_size / self.zoom_level))
            for x1, y1, x2, y2 in self.drawn_segments:
                last_img_x, last_img_y = self._canvas_to_image_coords(x1, y1)
                img_x, img_y = self._canvas_to_image_coords(x2, y2)
                cv2.line(self.mask, (last_img_x, last_img_y), (img_x, img_y), mask_value, draw_brush_size)
        self.canvas.delete("drawing_line"); self.update_display()

    # -------------------------------------------------------------------------
    # Canvas View Controls (Zoom, Pan, Display)
    # -------------------------------------------------------------------------
    def update_display(self):
        if self.current_image is None: return
        h, w = self.current_image.shape[:2]
        display_w, display_h = int(w * self.zoom_level), int(h * self.zoom_level)
        display_image_data = self.current_image.copy()
        
        overlay = np.zeros_like(display_image_data, dtype=np.uint8)
        for ann in self.annotations:
            color_rgb = self.CLASS_COLORS[ann['class_id'] % len(self.CLASS_COLORS)]
            polygons_for_cv2 = [np.array(p, dtype=np.int32) for p in ann['polygons']]
            cv2.fillPoly(overlay, polygons_for_cv2, color_rgb)
        
        display_image_data = cv2.addWeighted(display_image_data, 0.6, overlay, 0.4, 0)

        # Draw active mask (brush/SAM/editing) on top
        if np.any(self.mask):
            active_mask_overlay = np.zeros_like(self.current_image)
            color = (128, 128, 128)
            active_mask_overlay[self.mask > 0] = color
            display_image_data = cv2.addWeighted(display_image_data, 0.7, active_mask_overlay, 0.3, 0)

        # Draw active polygon on top
        if np.any(self.polygon_mask):
            active_poly_overlay = np.zeros_like(self.current_image)
            active_poly_overlay[self.polygon_mask > 0] = (255, 255, 0) # Yellow
            display_image_data = cv2.addWeighted(display_image_data, 0.7, active_poly_overlay, 0.3, 0)
        
        if len(self.polygon_points) > 0:
            for point in self.polygon_points: cv2.circle(display_image_data, tuple(map(int, point)), radius=3, color=(255, 255, 0), thickness=-1)
            if len(self.polygon_points) > 1: cv2.polylines(display_image_data, [np.array(self.polygon_points, dtype=np.int32)], isClosed=False, color=(255, 255, 0), thickness=1)

        pil_image = Image.fromarray(cv2.resize(display_image_data, (display_w, display_h), interpolation=cv2.INTER_NEAREST))
        self.current_photo = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.current_photo)
        self.canvas.configure(scrollregion=(self.offset_x, self.offset_y, self.offset_x + display_w, self.offset_y + display_h))
        self._update_status(); self._update_mask_status()

    def fit_to_window(self):
        if self.current_image is None: return
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <=1: canvas_w, canvas_h = 1000, 700
        h, w = self.current_image.shape[:2]
        self.zoom_level = min(canvas_w / w, canvas_h / h)
        self.offset_x, self.offset_y = 0, 0; self.update_display()

    def _zoom_in(self): self._zoom(delta=120)
    def _zoom_out(self): self._zoom(delta=-120)
    def _zoom(self, event=None, delta=None):
        if self.current_image is None: return
        if delta is None: delta = event.delta
        mouse_x = self.canvas.canvasx(event.x) if event else self.canvas.winfo_width() / 2
        mouse_y = self.canvas.canvasy(event.y) if event else self.canvas.winfo_height() / 2
        zoom_factor = 1.1 if delta > 0 else 0.9; old_zoom = self.zoom_level
        self.zoom_level = max(0.1, min(10.0, self.zoom_level * zoom_factor))
        if old_zoom != self.zoom_level:
            zoom_ratio = self.zoom_level / old_zoom
            self.offset_x = mouse_x - (mouse_x - self.offset_x) * zoom_ratio
            self.offset_y = mouse_y - (mouse_y - self.offset_y) * zoom_ratio
            self.update_display()

    def _start_panning(self, event): self.panning, self.pan_start_x, self.pan_start_y = True, event.x, event.y
    def _pan(self, event):
        if not self.panning: return
        dx, dy = event.x - self.pan_start_x, event.y - self.pan_start_y
        self.offset_x += dx; self.offset_y += dy
        self.pan_start_x, self.pan_start_y = event.x, event.y; self.update_display()
    def _stop_panning(self, event): self.panning = False

    # -------------------------------------------------------------------------
    # Utility & Helper Methods
    # -------------------------------------------------------------------------
    def clear_mask(self):
        if self.mask is not None: self.mask.fill(0)
        self.polygon_points = []; self.editing_instance_index = None
        if self.polygon_mask is not None: self.polygon_mask.fill(0)
        self.update_display()

    def clear_all_annotations(self):
        if messagebox.askokcancel("Confirm", "Are you sure you want to delete all annotations for this image?"):
            self.annotations = []; self.clear_mask()

    def _update_status(self):
        if self.image_files:
            filename = self.image_files[self.current_index].name
            self.status_var.set(f"Image {self.current_index + 1}/{len(self.image_files)}: {filename} (Zoom: {int(self.zoom_level*100)}%)")

    def _update_mask_status(self):
        if self.editing_instance_index is not None:
            self.mask_status_var.set(f"Editing instance {self.editing_instance_index}. Press 'i' to save changes.")
        elif np.any(self.mask) or self.polygon_points:
            self.mask_status_var.set(f"Instances: {len(self.annotations)} (Active drawing)")
        else:
            self.mask_status_var.set(f"Instances saved: {len(self.annotations)}")

    def _canvas_to_image_coords(self, canvas_x, canvas_y): return int((canvas_x - self.offset_x) / self.zoom_level), int((canvas_y - self.offset_y) / self.zoom_level)
    
    def _load_mask_from_file(self):
        if not self.mask_dir: messagebox.showwarning("Warning", "Please select a mask directory first."); return
        if self._try_load_existing_mask(): messagebox.showinfo("Success", "Mask loaded."); self._save_instance()
        else: messagebox.showerror("Error", f"Could not find a mask for '{self.image_files[self.current_index].name}' in '{self.mask_dir}'.")

    def _try_load_existing_mask(self):
        if not self.mask_dir or not self.image_files: return False
        image_path = self.image_files[self.current_index]
        mask_path = next((p for p in [self.mask_dir / f"{image_path.stem}.png", self.mask_dir / f"predicted_{image_path.stem}.png"] if p.exists()), None)
        if not mask_path: return False
        try:
            loaded_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if loaded_mask is None: return False
            h, w = self.current_image.shape[:2]
            if loaded_mask.shape != (h, w): loaded_mask = cv2.resize(loaded_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            _, self.mask = cv2.threshold(loaded_mask, 1, 255, cv2.THRESH_BINARY)
            return True
        except Exception as e: print(f"Error loading mask {mask_path}: {e}"); return False

    def _try_run_yolo_model(self):
        if not self.yolo_model: return
        results = self.yolo_model.predict(self.current_image, conf=0.5)
        if not results or results[0].masks is None: return
        h, w = self.current_image.shape[:2]
        masks, classes = results[0].masks.data.cpu().numpy(), results[0].boxes.cls.cpu().numpy()
        for i, mask in enumerate(masks):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            simplified_polygons = []
            for c in contours:
                if cv2.contourArea(c) < 10: continue
                simplified_contour = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True)
                if len(simplified_contour) >= 3: simplified_polygons.append(simplified_contour.squeeze(axis=1).tolist())
            if simplified_polygons: self.annotations.append({"class_id": int(classes[i]), "polygons": simplified_polygons})
        self.update_display()

    def _initialize_coco_dataset(self):
        self.coco_data = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": []}
        self.image_id_counter = 1; self.annotation_id_counter = 1

def main():
    root = tk.Tk()
    app = ImageAnnotationTool(root)
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit? This will finalize the COCO annotation file if enabled."):
            app._finalize_coco(); root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
