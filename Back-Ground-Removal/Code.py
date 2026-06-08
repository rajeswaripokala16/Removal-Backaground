import cv2
import mediapipe as mp
import numpy as np
import time

class VirtualBackgroundRemover:
    def __init__(self, bg_type='blur', bg_color=(114, 177, 255), bg_image_path=None, mask_threshold=0.5, save_video=False, video_path='output_bg_removed.mp4'):
        self.bg_type = bg_type
        self.bg_color = bg_color
        self.bg_image = cv2.imread(bg_image_path) if bg_image_path else None
        self.mask_threshold = mask_threshold
        self.save_video = save_video
        self.video_path = video_path
        self.segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.output_writer = None

    def get_background(self, frame):
        if self.bg_type == 'remove':
            return np.zeros(frame.shape, dtype=np.uint8)
        elif self.bg_type == 'solid':
            return np.full(frame.shape, self.bg_color, dtype=np.uint8)
        elif self.bg_type == 'image' and self.bg_image is not None:
            return cv2.resize(self.bg_image, (frame.shape[1], frame.shape[0]))
        elif self.bg_type == 'blur':
            return cv2.GaussianBlur(frame, (55, 55), 0)
        else:
            return frame.copy()

    def apply_virtual_bg(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentor.process(frame_rgb)
        mask = results.segmentation_mask > self.mask_threshold
        mask = mask.astype(np.uint8) * 255  # For visualization and PNG alpha
        background = self.get_background(frame)
        composite = np.where(mask[..., None], frame, background)
        return composite, mask

    def overlay_face_boxes(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(frame_rgb)
        if results.detections:
            ih, iw, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0,255,0), 2)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        height, width = frame.shape[:2]
        prev_time = time.time()
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_writer = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))
        cv2.namedWindow('Virtual Background Remover')
        cv2.createTrackbar('Mask Threshold','Virtual Background Remover',int(self.mask_threshold * 100),100,self.update_threshold)

        print("Controls:\n'b' = blur background\n's' = solid background\n'i' = image background\n'r' = remove background (black)\n'p' = save transparent PNG\nESC to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            output, mask = self.apply_virtual_bg(frame)
            output = self.overlay_face_boxes(output)

            # FPS calculation
            curr_time = time.time()
            disp_fps = 1.0 / (curr_time - prev_time + 1e-8)
            prev_time = curr_time
            cv2.putText(output, f'FPS: {int(disp_fps)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)

            cv2.imshow('Virtual Background Remover', output)
            cv2.imshow('Segmentation Mask', mask)
            if self.save_video and self.output_writer:
                self.output_writer.write(output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('b'):
                self.bg_type = 'blur'
            elif key == ord('s'):
                self.bg_type = 'solid'
            elif key == ord('i'):
                self.bg_type = 'image'
            elif key == ord('r'):
                self.bg_type = 'remove'
            elif key == ord('p'):
                # Save transparent PNG with mask as alpha
                rgba = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)
                rgba[..., 3] = mask
                cv2.imwrite('person_only.png', rgba)
                print("Saved transparent PNG frame as person_only.png")
            elif key == 27:  # ESC to exit
                break

        cap.release()
        if self.output_writer:
            self.output_writer.release()
        cv2.destroyAllWindows()

    def update_threshold(self, val):
        self.mask_threshold = val / 100.0

if __name__ == '__main__':
    # Update bg_image_path with a valid image file as needed.
    remover = VirtualBackgroundRemover(
        bg_type='blur', 
        bg_color=(0,255,0), 
        bg_image_path='background.jpg',  # Set your custom background image file here!
        mask_threshold=0.5,
        save_video=False,
        video_path='output_bg_removed.mp4'
    )
    remover.run()
