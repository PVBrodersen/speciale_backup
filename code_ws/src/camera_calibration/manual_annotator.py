import cv2
import collections
import glob
import csv
# mid, corner, left, top, right, bot for annotation with respect to extrinsics
class ManualTracker:
    def __init__(self, filenames):
        self.filenames = filenames
        self.annotations = collections.defaultdict(list)
        self.outputpath = "/home/philip/uni/speciale/code_ws/Annotations/extrinsic/zed/"
        self.circles = []  # To keep track of drawn circles

    def main(self):
        cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback('image', self.draw_circle)
        is_active = True

        for idx, filename in enumerate(self.filenames):
            frame = cv2.imread(filename)
            if not is_active:
                break
            self.frame_idx = idx
            self.frame = frame
            self.processed_frame = frame.copy()  # Copy of the frame to reset drawing
            self.circles.clear()  # Clear circles list for each new image

            while is_active:
                # Display the current processed frame
                # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('image',3000,1500)
                cv2.imshow('image', self.processed_frame)

                k = cv2.waitKey(1) & 0xFF

                if k == ord('n'):  # Next image
                    cv2.imwrite(self.outputpath+str(idx)+'_annotated.jpg',self.processed_frame)
                    break
                elif k == ord('z'):  # Undo the last circle
                    self.undo_last_circle()
                elif k == 27:  # Escape key to exit
                    cv2.imwrite(self.outputpath+str(idx)+'_annotated.jpg',self.processed_frame)
                    is_active = False

        # Print the annotations
        for key, annotated_points in self.annotations.items():
            f = open(self.outputpath+str(key)+'_annotations.csv','w')
            writer = csv.writer(f)
            for point in annotated_points:
                print("%d\t%d\t%d" % (key, point[0], point[1]))
                writer.writerow((key,point[0],point[1]))
            f.close()

    # Mouse callback function
    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Draw circle on the processed frame
            cv2.circle(self.processed_frame, (x, y), 5, (255, 0, 0), 2)
            # Save the circle position (frame index, x, y)
            self.circles.append((self.frame_idx, x, y))
            # Save annotation
            self.annotations[self.frame_idx].append((x, y))

    # Undo last circle
    def undo_last_circle(self):
        if self.circles:
            # Remove the last drawn circle
            last_frame_idx, x, y = self.circles.pop()
            # Remove from annotations
            if self.annotations[last_frame_idx]:
                self.annotations[last_frame_idx].pop()

            # Redraw the frame without the last circle
            self.processed_frame = self.frame.copy()
            for _, px, py in self.circles:
                cv2.circle(self.processed_frame, (px, py), 5, (255, 0, 0), 2)

def main(filenames):
    mt = ManualTracker(filenames=filenames)
    mt.main()

main(sorted(glob.glob('/home/philip/uni/speciale/code_ws/Images/zed*.jpg')))

