import cv2 as cv

# Function to check nested rectangles, i represents possible inner rectangle and o possible outer rectangle
def is_inside(i, o):
    ix, iy, iw, ih = i
    ox, oy, ow, oh = o

    return ix > ox and ix + iw < ox + ow and \
        iy > oy and iy + ih < oy + oh

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

img = cv.imread(r'Advance_Scripts\images\Lingotes.jpg')
found_rects, found_weights = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05, hitThreshold=0)

# Removing nested rectangles
found_rects_filtered = []
found_weights_filtered = []
for ri, r in enumerate(found_rects):
    for qi, q in enumerate(found_rects):
        if ri != qi and is_inside(r, q):
            break
    else:
        found_rects_filtered.append(r)
        found_weights_filtered.append(found_weights[ri])

# Highlighting detected people
for ri, r in enumerate(found_rects_filtered):
    x, y, w, h = r
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    text = '%.2f' % found_weights_filtered[ri]
    cv.putText(img, text, (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv.imshow('Lingotes detected', img)
cv.waitKey()
cv.destroyAllWindows()