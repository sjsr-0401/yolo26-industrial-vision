"""
데이터셋을 YOLO format으로 변환
1. NEU-DET: Pascal VOC XML -> YOLO txt
2. PCB Defect: 구조 확인 후 변환
3. Safety Helmet: Pascal VOC XML -> YOLO txt
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(r"C:\dev\active\yolo26-industrial-vision\datasets")


def convert_voc_to_yolo(xml_dir, img_dirs, out_dir, split_name, class_names):
    """
    Pascal VOC XML -> YOLO txt 변환
    xml_dir: XML 어노테이션 폴더
    img_dirs: 이미지 폴더 (서브폴더 포함 가능)
    out_dir: 출력 base 디렉토리
    """
    out_imgs = out_dir / split_name / "images"
    out_lbls = out_dir / split_name / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)
    
    # Build image lookup (filename -> path)
    img_lookup = {}
    for img_dir in img_dirs if isinstance(img_dirs, list) else [img_dirs]:
        for ext in ["*.jpg", "*.png", "*.bmp", "*.jpeg"]:
            for img in Path(img_dir).rglob(ext):
                img_lookup[img.name] = img
    
    xml_files = list(Path(xml_dir).glob("*.xml"))
    converted = 0
    skipped = 0
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get filename
            fname_elem = root.find("filename")
            if fname_elem is None:
                skipped += 1
                continue
            filename = fname_elem.text
            
            # Find image
            if filename not in img_lookup:
                # Try with different extensions
                stem = Path(filename).stem
                found = False
                for ext in [".jpg", ".png", ".bmp"]:
                    if stem + ext in img_lookup:
                        filename = stem + ext
                        found = True
                        break
                if not found:
                    skipped += 1
                    continue
            
            img_path = img_lookup[filename]
            
            # Get image size
            size = root.find("size")
            if size is None:
                from PIL import Image
                with Image.open(img_path) as im:
                    w, h = im.size
            else:
                w = int(size.find("width").text)
                h = int(size.find("height").text)
            
            if w == 0 or h == 0:
                skipped += 1
                continue
            
            # Convert objects
            labels = []
            for obj in root.findall("object"):
                name = obj.find("name").text.strip()
                
                # Map class name
                if name not in class_names:
                    # Try case-insensitive
                    name_lower = name.lower().replace("-", "_").replace(" ", "_")
                    matched = False
                    for i, cn in enumerate(class_names):
                        if cn.lower().replace("-", "_") == name_lower:
                            cls_id = i
                            matched = True
                            break
                    if not matched:
                        continue
                else:
                    cls_id = class_names.index(name)
                
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                # Clamp
                xmin = max(0, min(xmin, w))
                ymin = max(0, min(ymin, h))
                xmax = max(0, min(xmax, w))
                ymax = max(0, min(ymax, h))
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                # YOLO format: cx cy w h (normalized)
                cx = (xmin + xmax) / 2.0 / w
                cy = (ymin + ymax) / 2.0 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                
                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
            if labels:
                # Copy image
                dst_img = out_imgs / img_path.name
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
                
                # Write label
                lbl_path = out_lbls / (img_path.stem + ".txt")
                lbl_path.write_text("\n".join(labels), encoding="utf-8")
                converted += 1
            else:
                skipped += 1
                
        except Exception as e:
            skipped += 1
    
    print(f"    {split_name}: {converted} converted, {skipped} skipped")
    return converted


def setup_neu_det():
    """NEU-DET: VOC XML -> YOLO"""
    print("\n[1/3] NEU-DET Conversion")
    
    raw = BASE / "neu-raw" / "NEU-DET"
    out = BASE / "neu-det"
    
    if not raw.exists():
        print("  Raw data not found, skipping")
        return False
    
    # Remove old synthetic
    if out.exists():
        shutil.rmtree(out)
    
    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    
    # Train
    convert_voc_to_yolo(
        raw / "train" / "annotations",
        raw / "train" / "images",
        out, "train", class_names
    )
    
    # Val
    convert_voc_to_yolo(
        raw / "validation" / "annotations",
        raw / "validation" / "images",
        out, "val", class_names
    )
    
    # data.yaml
    yaml = f"""path: {str(out).replace(chr(92), '/')}
train: train/images
val: val/images

nc: {len(class_names)}
names: {class_names}
"""
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"  data.yaml created")
    return True


def setup_pcb():
    """PCB Defect: 구조 확인 후 변환"""
    print("\n[2/3] PCB Defect Conversion")
    
    raw = BASE / "pcb-raw"
    out = BASE / "pcb-defect"
    
    if not raw.exists():
        print("  Raw data not found, skipping")
        return False
    
    # Check structure
    all_files = list(raw.rglob("*"))
    xmls = [f for f in all_files if f.suffix == ".xml"]
    imgs = [f for f in all_files if f.suffix.lower() in [".jpg", ".png", ".bmp", ".jpeg"]]
    txts = [f for f in all_files if f.suffix == ".txt" and f.name != "classes.txt"]
    
    print(f"  Found: {len(imgs)} images, {len(xmls)} xml, {len(txts)} txt")
    
    # Check for classes.txt
    classes_file = None
    for f in raw.rglob("classes.txt"):
        classes_file = f
        break
    
    if classes_file:
        class_names = classes_file.read_text(encoding="utf-8").strip().split("\n")
        class_names = [c.strip() for c in class_names if c.strip()]
        print(f"  Classes from file: {class_names}")
    else:
        class_names = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
    
    if out.exists():
        shutil.rmtree(out)
    
    if xmls:
        # Find image directories
        img_dirs = list(set(i.parent for i in imgs))
        xml_dirs = list(set(x.parent for x in xmls))
        
        # Group by directory for train/val split
        import random
        random.seed(42)
        
        all_xml = sorted(xmls)
        random.shuffle(all_xml)
        split = int(len(all_xml) * 0.8)
        
        # Temporary: put all in one dir, convert, then split
        temp_xml = raw / "_temp_xml"
        temp_xml.mkdir(exist_ok=True)
        
        for x in all_xml[:split]:
            shutil.copy2(x, temp_xml / x.name)
        convert_voc_to_yolo(temp_xml, img_dirs, out, "train", class_names)
        shutil.rmtree(temp_xml)
        
        temp_xml.mkdir(exist_ok=True)
        for x in all_xml[split:]:
            shutil.copy2(x, temp_xml / x.name)
        convert_voc_to_yolo(temp_xml, img_dirs, out, "val", class_names)
        shutil.rmtree(temp_xml)
        
    elif txts:
        # Already YOLO format txt labels
        import random
        random.seed(42)
        
        # Match txt to images
        img_map = {i.stem: i for i in imgs}
        pairs = [(img_map[t.stem], t) for t in txts if t.stem in img_map]
        random.shuffle(pairs)
        split = int(len(pairs) * 0.8)
        
        for split_name, subset in [("train", pairs[:split]), ("val", pairs[split:])]:
            (out / split_name / "images").mkdir(parents=True, exist_ok=True)
            (out / split_name / "labels").mkdir(parents=True, exist_ok=True)
            for img, lbl in subset:
                shutil.copy2(img, out / split_name / "images" / img.name)
                shutil.copy2(lbl, out / split_name / "labels" / lbl.name)
        
        print(f"    train: {split}, val: {len(pairs)-split}")
    
    # data.yaml
    yaml = f"""path: {str(out).replace(chr(92), '/')}
train: train/images
val: val/images

nc: {len(class_names)}
names: {class_names}
"""
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    return True


def setup_safety():
    """Safety Helmet: VOC XML -> YOLO"""
    print("\n[3/3] Safety Helmet Conversion")
    
    raw = BASE / "safety-raw"
    out = BASE / "safety-helmet"
    
    if not raw.exists():
        print("  Raw data not found, skipping")
        return False
    
    # Check structure
    all_files = list(raw.rglob("*"))
    xmls = [f for f in all_files if f.suffix == ".xml"]
    imgs = [f for f in all_files if f.suffix.lower() in [".jpg", ".png", ".bmp", ".jpeg"]]
    
    print(f"  Found: {len(imgs)} images, {len(xmls)} xml")
    
    if out.exists():
        shutil.rmtree(out)
    
    # Detect class names from XMLs
    detected_classes = set()
    for xml_file in xmls[:50]:
        try:
            tree = ET.parse(xml_file)
            for obj in tree.getroot().findall("object"):
                name = obj.find("name").text.strip()
                detected_classes.add(name)
        except:
            pass
    
    print(f"  Detected classes: {sorted(detected_classes)}")
    class_names = sorted(detected_classes)
    
    if not class_names:
        class_names = ["helmet", "head", "person"]
    
    import random
    random.seed(42)
    
    all_xml = sorted(xmls)
    random.shuffle(all_xml)
    split = int(len(all_xml) * 0.8)
    
    img_dirs = list(set(i.parent for i in imgs))
    
    temp_xml = raw / "_temp_xml"
    
    temp_xml.mkdir(exist_ok=True)
    for x in all_xml[:split]:
        shutil.copy2(x, temp_xml / x.name)
    convert_voc_to_yolo(temp_xml, img_dirs, out, "train", class_names)
    shutil.rmtree(temp_xml)
    
    temp_xml.mkdir(exist_ok=True)
    for x in all_xml[split:]:
        shutil.copy2(x, temp_xml / x.name)
    convert_voc_to_yolo(temp_xml, img_dirs, out, "val", class_names)
    shutil.rmtree(temp_xml)
    
    yaml = f"""path: {str(out).replace(chr(92), '/')}
train: train/images
val: val/images

nc: {len(class_names)}
names: {class_names}
"""
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  Dataset Conversion to YOLO Format")
    print("=" * 60)
    
    setup_neu_det()
    setup_pcb()
    setup_safety()
    
    # Summary
    print("\n" + "=" * 60)
    print("  Final Dataset Summary")
    print("=" * 60)
    
    for d_name in ["neu-det", "pcb-defect", "safety-helmet"]:
        d = BASE / d_name
        if not d.exists():
            print(f"  {d_name}: NOT FOUND")
            continue
        
        for split in ["train", "val"]:
            imgs_dir = d / split / "images"
            lbls_dir = d / split / "labels"
            n_imgs = len(list(imgs_dir.glob("*"))) if imgs_dir.exists() else 0
            n_lbls = len(list(lbls_dir.glob("*"))) if lbls_dir.exists() else 0
            print(f"  {d_name}/{split}: {n_imgs} images, {n_lbls} labels")
        
        yaml = d / "data.yaml"
        if yaml.exists():
            content = yaml.read_text(encoding="utf-8")
            # Extract nc
            for line in content.split("\n"):
                if line.startswith("nc:"):
                    print(f"    classes: {line.strip()}")
