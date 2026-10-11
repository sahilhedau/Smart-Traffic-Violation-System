import requests
import time
import json
from pathlib import Path

def main():
    base_url = "http://127.0.0.1:8000"
    sample_video = Path("data/samples/sample_traffic.mp4")
    assert sample_video.exists(), "Sample video missing"

    print("[*] Uploading sample traffic video to server...")
    with open(sample_video, "rb") as f:
        files = {"file": ("test_run_traffic.mp4", f, "video/mp4")}
        upload_res = requests.post(f"{base_url}/api/video/upload", files=files)

    if upload_res.status_code != 200:
        print(f"[!] Upload failed: {upload_res.status_code} {upload_res.text}")
        return

    job_data = upload_res.json()
    job_id = job_data["job_id"]
    print(f"[*] Upload successful. Job ID: {job_id}")

    print("[*] Initiating video processing background worker...")
    proc_res = requests.post(f"{base_url}/api/video/process/{job_id}")
    if proc_res.status_code != 200:
        print(f"[!] Processing trigger failed: {proc_res.status_code} {proc_res.text}")
        return

    print("[*] Monitoring progress...")
    while True:
        status_res = requests.get(f"{base_url}/api/video/status/{job_id}")
        data = status_res.json()
        status = data["status"]
        progress = data["progress"]
        cur = data["current_frame"]
        total = data["total_frames"]
        fps = data["fps"]
        viols = data["violations_count"]

        print(f" -> Status: {status} | Progress: {progress:.1f}% | Frame: {cur}/{total} | FPS: {fps:.1f} | Violations Detected: {viols}")

        if status in ["COMPLETED", "FAILED"]:
            break
        time.sleep(2)

    print(f"\n[*] Processing ended with status: {status}")
    print(f"[*] Annotated video output saved at: {data.get('annotated_video_path')}")

    # Fetch updated statistics
    stats_res = requests.get(f"{base_url}/api/statistics")
    stats = stats_res.json()
    print("\n================ SYSTEM STATISTICS ================")
    print(f" Total Violations in DB : {stats['total_violations']}")
    print(f" Red Light Signal Jumps : {stats['red_light_count']}")
    print(f" Triple Riding Violations: {stats['triple_riding_count']}")
    print(f" No Helmet Violations   : {stats['no_helmet_count']}")
    print(f" Pending Manual Review  : {stats['pending_review_count']}")
    print(f" Confirmed by Reviewer  : {stats['confirmed_count']}")
    print("===================================================\n")

    # Fetch recent events
    events_res = requests.get(f"{base_url}/api/events?page=1&page_size=10")
    events = events_res.json()
    print("[*] Recent Violations Logged:")
    for ev in events["items"][:5]:
        print(f"  • [{ev['event_type']}] ID: {ev['id'][:8]} | Plate: {ev['plate_text']} | Confidence: {ev['confidence']:.2f} | Status: {ev['status']}")
        if ev.get("evidence_path"):
            print(f"    Evidence: {ev['evidence_path']}")

if __name__ == "__main__":
    main()
