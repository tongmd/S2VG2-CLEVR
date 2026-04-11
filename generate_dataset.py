import json
import random
import requests
import time
import os
import re
import sys

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Set GEMINI_API_KEY in the environment before running this script.")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={API_KEY}"

COLORS = ["red", "green", "blue", "yellow", "purple", "cyan", "magenta", "orange"]
SHAPES = ["circle", "rect", "triangle"]
SIZES = ["small", "large"]

def generate_scene(num_objects=5):
    objects = []
    svg_elements = []
    
    for _ in range(num_objects):
        c = random.choice(COLORS)
        s = random.choice(SHAPES)
        sz = random.choice(SIZES)
        
        # Dimensions based on size
        r = 10 if sz == "small" else 25
        w = h = 20 if sz == "small" else 50
        
        x = random.randint(30, 270)
        y = random.randint(30, 270)
        
        obj = {"id": len(objects), "color": c, "shape": s, "size": sz, "x": x, "y": y}
        objects.append(obj)
        
        if s == "circle":
            svg_elements.append(f'  <circle cx="{x}" cy="{y}" r="{r}" fill="{c}" />')
        elif s == "rect":
            svg_elements.append(f'  <rect x="{x-w//2}" y="{y-h//2}" width="{w}" height="{h}" fill="{c}" />')
        elif s == "triangle":
            # approximate an equilateral triangle
            svg_elements.append(f'  <polygon points="{x},{y-r} {x-r},{y+r} {x+r},{y+r}" fill="{c}" />')

    svg = f'<svg width="300" height="300">\n' + "\n".join(svg_elements) + '\n</svg>'
    return objects, svg

def generate_complex_benchmark(size=50):
    dataset = []
    random.seed(1024)
    for i in range(size):
        objs, svg = generate_scene(random.randint(4, 7))
        
        # We generate 3 types of questions per scene to test deep reasoning
        q_types = ["counting", "spatial_relation", "attribute_query"]
        q_type = random.choice(q_types)
        
        if q_type == "counting":
            target_c = random.choice(COLORS)
            target_s = random.choice(SHAPES)
            count = sum(1 for o in objs if o["color"] == target_c and o["shape"] == target_s)
            question = f"How many {target_c} {target_s}s are there?"
            answer = str(count)
            
        elif q_type == "spatial_relation":
            leftmost = min(objs, key=lambda o: o["x"])
            question = "What is the color of the object furthest to the left in the entire image?"
            answer = leftmost["color"]
                
        elif q_type == "attribute_query":
            target_sz = random.choice(SIZES)
            target_s = random.choice(SHAPES)
            matching = [o for o in objs if o["size"] == target_sz and o["shape"] == target_s]
            if matching:
                o = matching[0]
                question = f"Consider the {target_sz} {target_s} located around x={o['x']}, y={o['y']}. What is its color?"
                answer = o["color"]
            else:
                o = random.choice(objs)
                question = f"What is the size of the {o['color']} {o['shape']} located at approximately x={o['x']}, y={o['y']}?"
                answer = o["size"]
                
        dataset.append({
            "id": f"CLEVR-SVG-{i:03d}",
            "svg": svg,
            "question": question,
            "gt_answer": answer,
            "type": q_type,
            "raw_objects": objs
        })
    return dataset

print("Generating CLEVR-SVG Benchmark Dataset...")
dataset = generate_complex_benchmark(20) # Generate 20 questions for faster turnaround

with open("/home/stone/.openclaw/workspace/repos/cvpr_svg/CLEVR_SVG_Dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Running evaluation on Gemini 3.1 Pro Preview...")
results_log = []
correct = 0

for idx, item in enumerate(dataset):
    prompt = f"You are an expert visual reasoning agent. You are provided with the SVG code of an image.\n\nSVG:\n{item['svg']}\n\nQuestion: {item['question']}\n\nReason step-by-step about the coordinates and properties of the elements, then output your final answer on a new line starting with 'Final Answer: ' followed by the exact word or number (e.g. 'red', '2', 'circle')."
    
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.0}}
    
    try:
        resp = requests.post(URL, headers={'Content-Type': 'application/json'}, json=payload, timeout=10)
        resp_data = resp.json()
        if 'candidates' in resp_data:
            full_text = resp_data['candidates'][0]['content']['parts'][0]['text'].strip()
            match = re.search(r'Final Answer:\s*(.*)', full_text, re.IGNORECASE)
            if match:
                pred = match.group(1).strip().lower().replace('.', '')
            else:
                pred = full_text.split()[-1].strip().lower().replace('.', '')
        else:
            full_text = "API Error"
            pred = "error"
    except Exception as e:
        full_text = str(e)
        pred = "error"
        
    is_correct = (pred == str(item['gt_answer']).lower())
    if is_correct:
        correct += 1
        
    res_entry = {
        "id": item["id"],
        "question": item["question"],
        "gt_answer": item["gt_answer"],
        "predicted": pred,
        "is_correct": is_correct,
        "reasoning_trace": full_text
    }
    results_log.append(res_entry)
    print(f"[{idx+1}/20] GT: {item['gt_answer']:<8} | Pred: {pred:<8} | {'✅' if is_correct else '❌'}", flush=True)
    time.sleep(1.0)

with open("/home/stone/.openclaw/workspace/repos/cvpr_svg/CLEVR_SVG_Results.json", "w") as f:
    json.dump({"accuracy": correct / len(dataset), "total": len(dataset), "details": results_log}, f, indent=2)

print(f"\nEvaluation Complete! Accuracy: {correct}/{len(dataset)} ({correct/len(dataset)*100:.1f}%)")
