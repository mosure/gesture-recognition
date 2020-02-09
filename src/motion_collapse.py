import cv2


vs = cv2.VideoCapture(0)

back_sub = cv2.createBackgroundSubtractorMOG2(
    history=50,  # 500
    varThreshold=30,  # 16
    detectShadows=False  # True
)

temporal_frame = None

while True:
    check, frame = vs.read()

    if frame is None:
        break

    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    fg_mask = back_sub.apply(frame)

    bgr_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    b, g, r = cv2.split(bgr_mask)

    b *= 0

    bgr_mask = cv2.merge((b, g, r))

    if temporal_frame is None:
        temporal_frame = bgr_mask
        continue

    b, g, r = cv2.split(temporal_frame)

    # Set to g -= 1 for weird effect
    g[g > 0] -= 1
    r[g == 0] *= 0

    temporal_frame = cv2.merge((b, g, r))

    temporal_frame = cv2.addWeighted(temporal_frame, 0.5, bgr_mask, 0.5, 0)

    cv2.imshow('Mask', temporal_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
