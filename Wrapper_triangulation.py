import cv2
import numpy as np
import dlib
from imutils import face_utils
import scipy.spatial as spatial
import scipy.interpolate as interpolate

def ExtractFeatures(img,detector,predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if np.shape(rects)[0] == 0:
        shape = 0
        feature_found = False
    else:
        feature_found = True
        for (i, rect) in enumerate(rects):
            shape = predictor(img, rect)
            shape = face_utils.shape_to_np(shape)
    return shape, feature_found

def DrawDelaunayTriangles(img, subdiv, color):
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList :
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, color, 1, cv2.LINE_AA, 0)
    return img

def ExtractFace(img,points):
    allowance=10
    img_w, img_h = img.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)
    x, y = max(0, left-allowance), max(0, top-allowance)
    w, h = min(right+allowance, img_h)-x, min(bottom+allowance, img_w)-y
    points_cropped = points - np.asarray([[x, y]])
    rect = (x, y, w, h)
    cropped_img = img[y:y+h, x:x+w]
    return points_cropped, rect, cropped_img

def GenerateCoordinates(points):
    return np.asarray([(x,y) for y in range(np.min(points[:,1]) , np.max(points[:,1])+1) for x in range(np.min(points[:,0]) , np.max(points[:,0])+1)])

def Interpolation(image,cord):
    new_cord = np.int32(cord)
    x,y = new_cord
    dx ,dy = cord - new_cord
    bottom = image[y,x+1].T * dx + image[y,x].T * (1-dx)
    top = image[y+1,x+1].T * dx + image[y+1,x].T * (1-dx)
    return (top * dy + bottom * (1 - dy)).T

def WarpTriangles(src_img, result_img, tri_affines, dst_points, delaunay):
    roi = GenerateCoordinates(dst_points)
    roi_index = delaunay.find_simplex(roi)
    for index in range(len(delaunay.simplices)):
        coords = roi[roi_index == index]
        out_coords = np.dot(tri_affines[index],np.vstack((coords.T, np.ones(len(coords)))))
        x, y = coords.T
        result_img[y, x] = Interpolation(src_img, out_coords)
    return result_img

def ExtractTriangleMatrix(triangles, src_points, tar_points):
    one_vector = np.ones((3,), dtype=int)
    for triangle in triangles:
        src_tri = np.vstack((src_points[triangle, :].T, one_vector))
        tar_tri = np.vstack((tar_points[triangle, :].T, one_vector))
        matrix = np.dot(src_tri, np.linalg.inv(tar_tri))[:2, :]
        yield matrix

def FaceWarping(src_img, src_points, tar_points, tar_shape, ):
    w, h = tar_shape[:2]
    output = np.zeros((w,h, 3), dtype=np.uint8)
    delaunay = spatial.Delaunay(tar_points)
    # print("Delaunay Simplices ",len(delaunay.simplices))
    delaunay_triangles = np.asarray(list(ExtractTriangleMatrix(delaunay.simplices, src_points, tar_points)))
    output = WarpTriangles(src_img, output, delaunay_triangles, tar_points, delaunay)
    return output

def BlendingImages(img1, img2, points):
    frac = 0.75
    l = list(range(42, 48))
    r = list(range(36, 42))
    blur = frac * np.linalg.norm(np.mean(points[l], axis=0) - np.mean(points[r], axis=0))
    blur = int(blur)
    if blur % 2 == 0:
        blur += 1
    img1_blur = cv2.GaussianBlur(img1, (blur, blur), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur, blur), 0)
    img2_blur = img2_blur.astype(int)
    img2_blur += 128*(img2_blur <= 1)
    result = img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def SwapFace(src_img, tar_img,detector,predictor):
    src_points, flag1 = ExtractFeatures(src_img.copy(),detector,predictor)
    tar_points, flag2 = ExtractFeatures(tar_img.copy(),detector,predictor)

    if(flag1==False or flag2==False):
        check = False
        output = 0
        return output,check
    else:
        check = True
    size = np.shape(src_img)
    src_shape = (0,0,size[1],size[0])
    size = np.shape(tar_img)
    tar_img_rect = (0,0,size[1],size[0])

    src_img_subdiv  = cv2.Subdiv2D(src_shape)
    tar_img_subdiv  = cv2.Subdiv2D(tar_img_rect)

    src_points_tuple = tuple(map(tuple, src_points))

    for point in src_points_tuple:
        src_img_subdiv.insert(point)

    show_img = DrawDelaunayTriangles(src_img.copy(), src_img_subdiv,(0, 0, 255));
    cv2.imwrite("show_img_src.jpg",show_img)

    tar_points_tuple = tuple(map(tuple, tar_points))
    for point in tar_points_tuple:
        tar_img_subdiv.insert(point)
    show_img = DrawDelaunayTriangles(tar_img.copy(), tar_img_subdiv,(255, 0, 0));
    cv2.imwrite("show_img_tar.jpg",show_img)

    src_points, src_shape, src_face = ExtractFace(src_img.copy(), src_points)
    tar_points, tar_shape, tar_face = ExtractFace(tar_img.copy(), tar_points)

    w, h = tar_face.shape[:2]
    warped_src_face = FaceWarping(src_face.copy(), src_points, tar_points, (w, h))
    mask = np.zeros((w, h), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(tar_points), 255)
    mask = np.asarray(mask*(np.mean(warped_src_face, axis=2) > 0), dtype=np.uint8)
    # cv2.imwrite("warped_src_face.jpg",warped_src_face)

    warped_src_face = cv2.bitwise_and(warped_src_face,warped_src_face,mask=mask)
    tar_face_masked = cv2.bitwise_and(tar_face,tar_face,mask=mask)
    warped_src_face = BlendingImages(tar_face_masked, warped_src_face, tar_points)

    cv2.imwrite("warped_src_face.jpg",warped_src_face)
    r = cv2.boundingRect(mask)
    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
    output = cv2.seamlessClone(warped_src_face, tar_face, mask, center, cv2.NORMAL_CLONE)
    x, y, w, h = tar_shape
    result = tar_img.copy()
    result[y:y+h, x:x+w] = output
    return result, check


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("info.dat")

    img_src = cv2.imread('Data/Scarlett.jpg')

    cap = cv2.VideoCapture('Data/Test3.mp4')

    i=1
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('Test3OutputTri.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 6, (frame_width,frame_height))
    while(True):
        ret,img_tar = cap.read()
        if(not ret):
            break
        cv2.imwrite("Target_face.jpg",img_tar)
        output,flag = SwapFace(img_src , img_tar , detector , predictor)
        if(flag == False):
            out.write(img_tar)
            print i,"th Frame"
            i+=1
            continue
        out.write(output)
        print i,"th Frame"
        i+=1
    out.release()

if __name__ == '__main__':
    main()
