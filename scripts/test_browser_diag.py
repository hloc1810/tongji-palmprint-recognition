"""Comprehensive Playwright verification script for Tongji Palmprint UI."""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        print("==================================================")
        print("TEST 1: http://127.0.0.1:8000/ (Web server mode)")
        print("==================================================")
        page1 = browser.new_page()
        page1_logs = []
        page1.on("console", lambda msg: page1_logs.append(f"[{msg.type}] {msg.text}"))
        page1.on("pageerror", lambda exc: page1_logs.append(f"[PAGE ERROR] {exc}"))
        page1.on("response", lambda res: print(f"[4xx/5xx] {res.status} {res.url}") if res.status >= 400 else None)

        res = page1.goto("http://127.0.0.1:8000/")
        page1.wait_for_timeout(1000)
        assert res.status == 200, f"Expected 200, got {res.status}"
        print("Title:", page1.title())
        print("Mode pill:", page1.locator("#modeText").inner_text().strip())

        # Test Sample 1 recognize
        page1.locator("#recognizeButton").click()
        page1.wait_for_timeout(2000)
        id_val = page1.locator("#identityValue").inner_text().strip()
        badge_val = page1.locator("#resultBadge").inner_text().strip()
        print(f"Sample 1 Recognize -> Identity: {id_val}, Badge: {badge_val}")
        assert "0001" in id_val, f"Expected Palm ID 0001, got {id_val}"

        # Test Model switch to Linear SVM
        print("Switching model to Linear SVM...")
        page1.select_option("#modelSelect", "svm")
        page1.wait_for_timeout(500)
        page1.locator("#recognizeButton").click()
        page1.wait_for_timeout(2000)
        id_svm = page1.locator("#identityValue").inner_text().strip()
        print(f"Linear SVM Recognize -> Identity: {id_svm}")
        assert "0001" in id_svm

        # Test Model switch to 1-NN
        print("Switching model to 1-NN...")
        page1.select_option("#modelSelect", "knn")
        page1.wait_for_timeout(500)
        page1.locator("#recognizeButton").click()
        page1.wait_for_timeout(2000)
        id_knn = page1.locator("#identityValue").inner_text().strip()
        print(f"1-NN Recognize -> Identity: {id_knn}")
        assert "0001" in id_knn

        # Test uploading an image
        print("Testing file upload with session2/00415.bmp (Palm ID 0042)...")
        page1.select_option("#modelSelect", "crc")
        file_input = page1.locator("#fileInput")
        file_input.set_input_files("D:/Nhandienmau/DA/data/raw/session2/00415.bmp")
        page1.wait_for_timeout(500)
        print("Filename shown:", page1.locator("#filename").inner_text().strip())
        page1.locator("#recognizeButton").click()
        page1.wait_for_timeout(2000)
        upload_id = page1.locator("#identityValue").inner_text().strip()
        upload_desc = page1.locator("#resultDescription").inner_text().strip()
        print(f"Uploaded file Recognize -> Identity: {upload_id}, Desc: {upload_desc}")
        assert "0042" in upload_id, f"Expected Palm ID 0042, got {upload_id}"

        # Check console logs for page1
        err_logs_1 = [log_line for log_line in page1_logs if "error" in log_line.lower()]
        print("Page 1 error logs count:", len(err_logs_1))
        if err_logs_1:
            for log_line in err_logs_1:
                print("  ", log_line)

        print("\n==================================================")
        print("TEST 2: file:///D:/Nhandienmau/DA/Tongji_Palmprint_UI.html (Direct file mode)")
        print("==================================================")
        page2 = browser.new_page()
        page2_logs = []
        page2.on("console", lambda msg: page2_logs.append(f"[{msg.type}] {msg.text}"))
        page2.on("pageerror", lambda exc: page2_logs.append(f"[PAGE ERROR] {exc}"))

        page2.goto("file:///D:/Nhandienmau/DA/Tongji_Palmprint_UI.html")
        page2.wait_for_timeout(1000)
        print("Mode pill (file://):", page2.locator("#modeText").inner_text().strip())

        # Test Sample Recognize from file protocol
        rec2_btn = page2.locator("#recognizeButton")
        rec2_btn.click()
        page2.wait_for_timeout(2000)
        id_val2 = page2.locator("#identityValue").inner_text().strip()
        toast2 = page2.locator("#toast").inner_text().strip()
        print(f"file:// Recognize -> Identity: {id_val2}, Toast: {toast2}")
        assert "0001" in id_val2, f"Expected Palm ID 0001, got {id_val2}"

        # Test sample switching on file protocol
        print("Switching to Sample 3 (Palm ID 0128)...")
        page2.locator("button[data-sample-idx='2']").click()
        page2.wait_for_timeout(500)
        rec2_btn.click()
        page2.wait_for_timeout(2000)
        id_val3 = page2.locator("#identityValue").inner_text().strip()
        print(f"file:// Sample 3 -> Identity: {id_val3}")
        assert "0128" in id_val3, f"Expected Palm ID 0128, got {id_val3}"

        err_logs_2 = [log_line for log_line in page2_logs if "error" in log_line.lower()]
        print("Page 2 error logs count:", len(err_logs_2))
        if err_logs_2:
            for log_line in err_logs_2:
                print("  ", log_line)

        browser.close()
        print("\nALL PLAYWRIGHT TESTS PASSED SUCCESFULLY!")

if __name__ == "__main__":
    main()
