import numpy as np
import argparse
import dlib
import cv2
import random
from scipy.spatial import ConvexHull, convex_hull_plot_2d

DEBUG = True

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def convex_contains(convex, pt):
    n = len(convex)
    if n < 3:
        return False

    v0 = np.cross(convex[0]-pt, convex[1]-pt)
    for i in range(n):
        v1 = np.cross(convex[i]-pt, convex[(i+1)%n]-pt)
        if v0*v1 < 0:
            return False
    return True

def convex_contains_test():
    convex = np.array([[1,1], [1,-1], [-1,-1], [-1,1]])
    pt = np.array([0,0])
    if convex_contains(convex, pt) == False:
        print("Fail 1")
    else:
        print("Success 1")

    pt = np.array([2,0])
    if convex_contains(convex, pt) == True:
        print("Fail 2")
    else:
        print("Success 2")

# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, (p[0], p[1]), 2, color, -1, cv2.LINE_AA, 0 )

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
        
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


# Draw voronoi diagram
def draw_voronoi(img, subdiv) :

    ( facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
        
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), -1, cv2.LINE_AA, 0)


def getFaceLandmarks(detector, predictor, img_gray):
    faces_in_image = detector(img_gray, 0)
    
    '''
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect);
    '''
    face_landmarks = []
    for face in faces_in_image:

        landmarks = predictor(img_gray, face)

        landmarks_list = []
        for i in range(0, landmarks.num_parts):
            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))
            
        '''
        win_delaunay = "Delaunay Triangulation"
        win_voronoi = "Voronoi Diagram"
        #for landmark_num, xy in enumerate(landmarks_list, start = 1):
        #    cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
        #    cv2.putText(img, str(landmark_num),(xy[0]-7,xy[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)

        # Insert points into subdiv
        for p in landmarks_list :
            subdiv.insert(p)
        
            img_copy = img.copy()
            # Draw delaunay triangles
            draw_delaunay( img_copy, subdiv, (255, 255, 255) );
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(1)

        # Draw delaunay triangles
        draw_delaunay( img, subdiv, (255, 255, 255) );

        # Draw points
        for p in landmarks_list :
            draw_point(img, p, (0,0,255))

        # Allocate space for Voronoi Diagram
        img_voronoi = np.zeros(img.shape, dtype = img.dtype)

        # Draw Voronoi diagram
        draw_voronoi(img_voronoi,subdiv)

        # Show results
        cv2.imshow(win_delaunay,img)
        cv2.imshow(win_voronoi,img_voronoi)
        cv2.waitKey(0)
        '''
        face_landmarks.append(landmarks_list)
    return np.array(face_landmarks)

def U(r):
    r_2 = r*r
    if r_2 == 0:
        return 0
    return r*r*np.log(r*r)

def getWxy(face_landmarks_src, ps_dst):
    p = len(ps_dst)
    A = np.zeros((p+3,p+3))
    bx = np.zeros((p+3,1))
    by = np.zeros((p+3,1))
    for i in range(p):
        for j in range(p):
            #A[i][j] = U(abs(ps_dst[i][0]-ps_dst[j][0]) + abs(ps_dst[i][1]-ps_dst[j][1]))
            A[i][j] = U(np.sqrt(np.square(ps_dst[i][0]-ps_dst[j][0]) + np.square(ps_dst[i][1]-ps_dst[j][1])))
        A[i][p] = ps_dst[i][0]
        A[i][p+1] = ps_dst[i][1]
        A[i][p+2] = 1

    for j in range(p):
        A[p][j] = ps_dst[i][0]
        A[p+1][j] = ps_dst[i][1]
        A[p+2][j] = 1
                
    lamda = 0.01
    for i in range(p+3):
        A[i][i] += lamda

    for i in range(p):
        bx[i] = face_landmarks_src[i][0]
        by[i] = face_landmarks_src[i][1]

    A_inv = np.linalg.inv(A)
    wx = np.matmul(A_inv, bx)
    wy = np.matmul(A_inv, by)
    
    wx = np.reshape(wx, (wx.shape[0]))
    wy = np.reshape(wy, (wy.shape[0]))

    for i in range(len(ps_dst)):
        bx[i] -= f(ps_dst[i][0], ps_dst[i][1], wx, ps_dst)
        by[i] -= f(ps_dst[i][0], ps_dst[i][1], wy, ps_dst)

    return wx, wy

def f(x,y,w,pts):
    res = w[-1] + w[-3]*x + w[-2]*y
    for i in range(len(pts)):
        #res += w[i] * U(abs(pts[i][0]-x) + abs(pts[i][1]-y))
        res += w[i] * U(np.sqrt(np.square(pts[i][0]-x) + np.square(pts[i][1]-y)))

    return res

def getFaceSize(face_landmark):
    x_min = int(np.min(face_landmark[:,0]))
    y_min = int(np.min(face_landmark[:,1]))
    x_max = int(np.max(face_landmark[:,0]))
    y_max = int(np.max(face_landmark[:,1]))
    area = (y_max - y_min) * (x_max - x_min)
    #print('area', area)
    return area

def getRemapValue(dst_img, src_img, ps_dst_ori, ps_src_ori):
    ps_dst = np.array(ps_dst_ori)
    ps_src = np.array(ps_src_ori)
    x_min_dst = int(np.min(ps_dst[:,0]))
    y_min_dst = int(np.min(ps_dst[:,1]))
    x_max_dst = int(np.max(ps_dst[:,0]))
    y_max_dst = int(np.max(ps_dst[:,1]))
    face_dst = dst_img[y_min_dst:y_max_dst, x_min_dst:x_max_dst]

    x_min_src = int(np.min(ps_src[:,0]))
    y_min_src = int(np.min(ps_src[:,1]))
    x_max_src = int(np.max(ps_src[:,0]))
    y_max_src = int(np.max(ps_src[:,1]))
    face_src = src_img[y_min_src:y_max_src, x_min_src:x_max_src]
    
    v_min_dst = np.min(face_dst)
    v_max_dst = np.max(face_dst)
    v_mean_dst = np.mean(face_dst)

    v_min_src = np.min(face_src)
    v_max_src = np.max(face_src)
    v_mean_src = np.mean(face_src)

    rescale = (v_max_dst - v_min_dst) / (v_max_src - v_min_src)
    offset = v_mean_dst - v_mean_src * rescale

    return rescale, offset

def replaceFace(dst_img, src_img, ps_dst, ps_src, wx, wy):
    hull = ConvexHull(ps_dst)
    convex = []
    for i in range(len(hull.vertices)):
        convex.append(hull.points[hull.vertices[i]])
    convex = np.array(convex)
    
    '''
    for i in range(len(convex)):
        p1 = (int(convex[i][0]), int(convex[i][1]))
        p2 = (int(convex[(i+1)%len(convex)][0]), int(convex[(i+1)%len(convex)][1]))
        #cv2.line(dst_img, p1, p2, (255,255,255))

    n = len(ps_dst)
    for i in range(n):
        pt = ps_dst[i]
        k=10
        for j in range(k):
            x = int(ps_dst[i][0]*(1-j/(float)(k)) + ps_dst[(i+1)%n][0]*(j/(float)(k)))
            y = int(ps_dst[i][1]*(1-j/(float)(k)) + ps_dst[(i+1)%n][1]*(j/(float)(k)))
            srcx = int(f(x,y,wx,ps_dst))
            srcy = int(f(x,y,wy,ps_dst))
            print(x, y, srcx, srcy, x-srcx, y-srcy)
            #dst_img[y,x] = src_img[srcy, srcx]
            dst_img[y,x] = (255,255,255)
            src_img[srcy, srcx] = (255,255,255)
                
            cv2.imshow('src', src_img)
            cv2.imshow('dst', dst_img)
            cv2.waitKey(0)
    '''
    
    convex = convex.astype(int)
    mask = np.zeros(dst_img.shape, dst_img.dtype)
    cv2.fillPoly(mask, [convex], (255, 255, 255))

    rescale, offset = getRemapValue(dst_img, src_img, ps_dst, ps_src)

    src_img_warped = np.zeros(dst_img.shape, dst_img.dtype)
    x_min = int(np.min(convex[:,0]))
    y_min = int(np.min(convex[:,1]))
    x_max = int(np.max(convex[:,0]))
    y_max = int(np.max(convex[:,1]))
    
    #cv2.imwrite('src_img.jpg', src_img)
    #cv2.imwrite('dst_img.jpg', dst_img)

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            #if convex_contains(convex, (x,y)):
            if mask[y,x][0] > 128:
                srcx = int(f(x,y,wx,ps_dst))
                srcy = int(f(x,y,wy,ps_dst))
                #print(x, y, srcx, srcy, x-srcx, y-srcy)
                v = src_img[srcy, srcx] * rescale + offset
                v = np.clip(v, 0, 255)
                src_img_warped[y,x] = v
                #dst_img[y,x] = v
                #src_img[srcy, srcx] = (255,255,255)
                
                #cv2.imshow('src', src_img)
                #cv2.imshow('dst', dst_img)
                #cv2.waitKey(0)
                

    border = 0
    h, w, c = src_img_warped.shape
    src_img_warped = src_img_warped[border:h-border, border:w-border]
    mask = mask[border:h-border, border:w-border]
    
    cv2.imwrite('test2_ori.jpg', dst_img)

    center = (int((x_min+x_max)/2), int((y_min+y_max)/2))
    #center = (int(dst_img.shape[1]/2), int(dst_img.shape[0]/2))
    dst_img = cv2.seamlessClone(src_img_warped, dst_img, mask, center, cv2.NORMAL_CLONE)
    
    cv2.imwrite('res2_tps.jpg', dst_img)
    cv2.imshow('res2_tps.jpg', dst_img)
    cv2.waitKey(0)

    return dst_img

def changeFace1(image_path, video_path, out_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("info.dat")

    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_landmarks_src = getFaceLandmarks(detector, predictor, img_gray)[0]
    
    img_copy = img.copy()
    for p in face_landmarks_src:
        draw_point(img_copy, p, (0,0,255))
        
    for landmark_num, xy in enumerate(face_landmarks_src, start = 1):
        cv2.circle(img_copy, (xy[0], xy[1]), 12, (168, 0, 20), -1)
        cv2.putText(img_copy, str(landmark_num),(xy[0]-7,xy[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)
        
    cv2.imshow('src img_copy',img_copy)
    cv2.waitKey(1)

    cap = cv2.VideoCapture(video_path)
    
    if(cap.isOpened()):
        ret, frame = cap.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (frame.shape[1],frame.shape[0]))

    face_landmark_buffer = []
    alpha = 0.5

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_landmarks_dst = getFaceLandmarks(detector, predictor, frame_gray)
            
            if len(face_landmarks_dst) == 0:
                frame_gray = cv2.equalizeHist(frame_gray)
                face_landmarks_dst = getFaceLandmarks(detector, predictor, frame_gray)

            if len(face_landmarks_dst) == 0:
                k = 4.0
                frame_gray = cv2.resize(frame_gray, (0,0), fx=k, fy=k)

                face_landmarks_dst = getFaceLandmarks(detector, predictor, frame_gray)
            
                frame_gray = cv2.resize(frame_gray, (0,0), fx=1/k, fy=1/k)
                for i in range(len(face_landmarks_dst)):
                    for j in range(len(face_landmarks_dst[i])):
                        #face_landmarks_dst[i][j] = (round(face_landmarks_dst[i][j][0]/k), round(face_landmarks_dst[i][j][1]/k))
                        face_landmarks_dst[i][j] = [round(face_landmarks_dst[i][j][0]/k), round(face_landmarks_dst[i][j][1]/k)]

            if DEBUG:
                frame_debug = frame.copy()
                for face_landmark in face_landmarks_dst :
                    for p in face_landmark:
                        draw_point(frame_debug, p, (0,0,255))
                cv2.imshow('frame_landmarks', frame_debug)
                
            if len(face_landmarks_dst) == 0:
                face_landmark_buffer = []

            for ps_dst in face_landmarks_dst:
                if len(face_landmark_buffer) == 0:
                    face_landmark_buffer = ps_dst
                else:
                    face_landmark_buffer = face_landmark_buffer*(1-alpha) + alpha*ps_dst
                    ps_dst = face_landmark_buffer

                wx, wy = getWxy(face_landmarks_src, ps_dst)
                frame = replaceFace(frame, img, ps_dst, face_landmarks_src, wx, wy)
            
            cv2.imshow('Frame',frame)
            cv2.imshow('frame_gray',frame_gray)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break

    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def changeFace2(video_path, out_path):
    #convex_contains_test()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("info.dat")

    cap = cv2.VideoCapture(video_path)
    
    if(cap.isOpened()):
        ret, frame = cap.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_path,fourcc, 20.0, (frame.shape[1],frame.shape[0]))
    
    face_landmark_buffer1 = []
    face_landmark_buffer2 = []
    alpha = 0.8

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            
            face_landmarks_dst = getFaceLandmarks(detector, predictor, frame)

            if len(face_landmarks_dst) < 2:
                k = 4.0
                frame = cv2.resize(frame, (0,0), fx=k, fy=k)

                face_landmarks_dst = getFaceLandmarks(detector, predictor, frame)
            
                frame = cv2.resize(frame, (0,0), fx=1/k, fy=1/k)
                for i in range(len(face_landmarks_dst)):
                    for j in range(len(face_landmarks_dst[i])):
                        #face_landmarks_dst[i][j] = (round(face_landmarks_dst[i][j][0]/k), round(face_landmarks_dst[i][j][1]/k))
                        face_landmarks_dst[i][j] = [round(face_landmarks_dst[i][j][0]/k), round(face_landmarks_dst[i][j][1]/k)]

                
            if len(face_landmarks_dst) < 2:
                face_landmark_buffer1 = []
                face_landmark_buffer2 = []
                
            frame_copy = frame.copy()
            if len(face_landmarks_dst) == 2:
                area_threshold = 1000
                if (getFaceSize(face_landmarks_dst[0]) < area_threshold or getFaceSize(face_landmarks_dst[1]) < area_threshold):
                    out.write(frame_copy)
                    continue

                if np.mean(face_landmarks_dst[0][:,0]) < np.mean(face_landmarks_dst[1][:,0]):
                    ps_dst1 = face_landmarks_dst[0]
                    ps_dst2 = face_landmarks_dst[1]
                else:
                    ps_dst1 = face_landmarks_dst[1]
                    ps_dst2 = face_landmarks_dst[0]


                if len(face_landmark_buffer1) == 0:
                    face_landmark_buffer1 = ps_dst1
                else:
                    face_landmark_buffer1 = face_landmark_buffer1*(1-alpha) + alpha*ps_dst1
                    ps_dst1 = face_landmark_buffer1.astype(int)

                if len(face_landmark_buffer2) == 0:
                    face_landmark_buffer2 = ps_dst2
                else:
                    face_landmark_buffer2 = face_landmark_buffer2*(1-alpha) + alpha*ps_dst2
                    ps_dst2 = face_landmark_buffer2.astype(int)
                    
                if DEBUG:
                    frame_debug = frame.copy()
                    for p in ps_dst1:
                        draw_point(frame_debug, p, (0,0,255))
                    for p in ps_dst2:
                        draw_point(frame_debug, p, (0,0,255))
                    cv2.imshow('frame_landmarks', frame_debug)
                    cv2.waitKey(1)

                
                wx, wy = getWxy(ps_dst2, ps_dst1)
                frame_copy = replaceFace(frame_copy, frame, ps_dst1, ps_dst2, wx, wy)

                wx, wy = getWxy(ps_dst1, ps_dst2)
                frame_copy = replaceFace(frame_copy, frame, ps_dst2, ps_dst1, wx, wy)
                
            
            cv2.imshow('Frame',frame_copy)
            out.write(frame_copy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break

    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def main():
    image_path = "Data/Rambo.jpg"
    video_path = "Data/Test1.mp4"
    out_path = "output1.mp4"
    #changeFace1(image_path, video_path, out_path)
    
    video_path = "Data/Test2.mp4"
    out_path = "output2.mp4"
    #changeFace2(video_path, out_path)
    
    image_path = "Data/Scarlett.jpg"
    video_path = "Data/Test3.mp4"
    out_path = "output3.mp4"
    changeFace1(image_path, video_path, out_path)


if __name__ == '__main__':
    main()