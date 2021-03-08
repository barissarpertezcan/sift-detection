import cv2

query_bayrak = cv2.imread("turkbayrak.jpg")
query_odtu = cv2.imread("odtulogo.jpg")
query_stm = cv2.imread("stmlogo.jpg")
query_ort = cv2.imread("ortlogo.jpg")

query_bayrak_gray = cv2.cvtColor(query_bayrak, cv2.COLOR_BGR2GRAY)
query_odtu_gray = cv2.cvtColor(query_odtu, cv2.COLOR_BGR2GRAY)
query_stm_gray = cv2.cvtColor(query_stm, cv2.COLOR_BGR2GRAY)
query_ort_gray = cv2.cvtColor(query_ort, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

query_keypoints_bayrak, query_descriptor_bayrak = sift.detectAndCompute(query_bayrak_gray, None)
query_keypoints_odtu, query_descriptor_odtu = sift.detectAndCompute(query_odtu_gray, None)
query_keypoints_stm, query_descriptor_stm = sift.detectAndCompute(query_stm_gray, None)
query_keypoints_ort, query_descriptor_ort = sift.detectAndCompute(query_ort_gray, None)
# deneme = cv2.drawKeypoints(training_image, train_keypoints, None, [0,255,0], 4)
# cv2.imwrite("keypoints.jpg", deneme)

cap = cv2.VideoCapture(0)

depo_a = [0, 0, 0, 0, 0]
depo_b = [0, 0, 0, 0, 0]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == 1:
        train_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # SIFT
        train_keypoints, train_descriptor = sift.detectAndCompute(train_gray, None)

        # BF MATCH
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches_bayrak = bf.match(query_descriptor_bayrak, train_descriptor)
        matches_odtu = bf.match(query_descriptor_odtu, train_descriptor)
        matches_stm = bf.match(query_descriptor_stm, train_descriptor)
        matches_ort = bf.match(query_descriptor_ort, train_descriptor)

        # ADJUST THRESHOLD WITH RESPECT TO IMAGES
        thresh_bayrak = 65
        thresh_odtu = 58
        thresh_stm = 55 #60
        thresh_ort = 85 #75

        bayrak = []
        odtu = []
        stm = []
        ort = []

        # GET BEST MATCHES
        for each in matches_bayrak:
            # print("each.distance\n\n", each.distance)
            if each.distance < thresh_bayrak:
                bayrak.append(each)

        for each in matches_odtu:
            if each.distance < thresh_odtu:
                odtu.append(each)

        for each in matches_stm:
            if each.distance < thresh_stm:
                stm.append(each)

        for each in matches_ort:
            if each.distance < thresh_ort:
                ort.append(each)

        bayrak_coordinate = []
        odtu_coordinate = []
        stm_coordinate = []
        ort_coordinate = []

        # GET PIXEL COORDINATES FROM BEST MATCHES
        for mat in bayrak:
            # Get the matching keypoints for each of the images
            bayrak_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x, y) = train_keypoints[bayrak_idx].pt

            # Append to each list
            bayrak_coordinate.append((x, y))

        for mat in odtu:
            # Get the matching keypoints for each of the images
            odtu_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x, y) = train_keypoints[odtu_idx].pt

            # Append to each list
            odtu_coordinate.append((x, y))

        for mat in stm:
            # Get the matching keypoints for each of the images
            stm_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x, y) = train_keypoints[stm_idx].pt

            # Append to each list
            stm_coordinate.append((x, y))

        for mat in ort:
            # Get the matching keypoints for each of the images
            ort_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x, y) = train_keypoints[ort_idx].pt

            # Append to each list
            ort_coordinate.append((x, y))

        list = [len(bayrak), len(odtu), len(stm), len(ort)]
        #print("list", list)

        totala = 0
        totalb = 0

        if list[0] > 10 or list[1] > 10 or list[2] > 10 or list[3] > 10:
            if max(list) == list[0]:
                for every in bayrak_coordinate:
                    a = every[0]
                    b = every[1]
                    totala += a
                    totalb += b
                ort_a = int(totala / len(bayrak_coordinate))
                ort_b = int(totalb / len(bayrak_coordinate))

                # STORE THE AVERAGE PIXEL COORDINATE IN A LIST
                depo_a[4] = depo_a[3]
                depo_a[3] = depo_a[2]
                depo_a[2] = depo_a[1]
                depo_a[1] = depo_a[0]
                depo_a[0] = ort_a

                depo_b[4] = depo_b[3]
                depo_b[3] = depo_b[2]
                depo_b[2] = depo_b[1]
                depo_b[1] = depo_b[0]
                depo_b[0] = ort_b

                # GET THE MEAN PIXEL LOCATION USING PREVIOUS AVERAGE PIXEL VALUES
                result_a = int(((depo_a[0] + depo_a[1] + depo_a[2] + depo_a[3] + depo_a[4]) / 5))
                result_b = int(((depo_b[0] + depo_b[1] + depo_b[2] + depo_b[3] + depo_b[4]) / 5))

                print("TÜRK BAYRAĞI")
                # MIDDLE AREA/COORDINATE DETECTION
                frame[result_b - 3:result_b + 4, result_a - 3:result_a + 4] = 0

            if max(list) == list[1]:
                for every in odtu_coordinate:
                    a = every[0]
                    b = every[1]
                    totala += a
                    totalb += b
                ort_a = int(totala / len(odtu_coordinate))
                ort_b = int(totalb / len(odtu_coordinate))
                # STORE THE AVERAGE PIXEL IN A LIST.
                depo_a[4] = depo_a[3]
                depo_a[3] = depo_a[2]
                depo_a[2] = depo_a[1]
                depo_a[1] = depo_a[0]
                depo_a[0] = ort_a

                depo_b[4] = depo_b[3]
                depo_b[3] = depo_b[2]
                depo_b[2] = depo_b[1]
                depo_b[1] = depo_b[0]
                depo_b[0] = ort_b

                # GET THE PIXEL LOCATION AVERAGING PREVIOUS PIXEL VALUES
                result_a = int(((depo_a[0] + depo_a[1] + depo_a[2] + depo_a[3] + depo_a[4]) / 5))
                result_b = int(((depo_b[0] + depo_b[1] + depo_b[2] + depo_b[3] + depo_b[4]) / 5))

                print("ODTÜ")
                # MIDDLE AREA/COORDINATE DETECTION
                frame[result_b - 3:result_b + 4, result_a - 3:result_a + 4] = 0

            if max(list) == list[2]:
                for every in stm_coordinate:
                    a = every[0]
                    b = every[1]
                    totala += a
                    totalb += b
                ort_a = int(totala / len(stm_coordinate))
                ort_b = int(totalb / len(stm_coordinate))
                # STORE THE AVERAGE PIXEL IN A LIST.
                depo_a[4] = depo_a[3]
                depo_a[3] = depo_a[2]
                depo_a[2] = depo_a[1]
                depo_a[1] = depo_a[0]
                depo_a[0] = ort_a

                depo_b[4] = depo_b[3]
                depo_b[3] = depo_b[2]
                depo_b[2] = depo_b[1]
                depo_b[1] = depo_b[0]
                depo_b[0] = ort_b

                # GET THE PIXEL LOCATION AVERAGING PREVIOUS PIXEL VALUES
                result_a = int(((depo_a[0] + depo_a[1] + depo_a[2] + depo_a[3] + depo_a[4]) / 5))
                result_b = int(((depo_b[0] + depo_b[1] + depo_b[2] + depo_b[3] + depo_b[4]) / 5))

                print("STM")
                # MIDDLE AREA/COORDINATE DETECTION
                frame[result_b - 3:result_b + 4, result_a - 3:result_a + 4] = 0

            if max(list) == list[3]:
                for every in ort_coordinate:
                    a = every[0]
                    b = every[1]
                    totala += a
                    totalb += b
                ort_a = int(totala / len(ort_coordinate))
                ort_b = int(totalb / len(ort_coordinate))
                # STORE THE AVERAGE PIXEL IN A LIST.
                depo_a[4] = depo_a[3]
                depo_a[3] = depo_a[2]
                depo_a[2] = depo_a[1]
                depo_a[1] = depo_a[0]
                depo_a[0] = ort_a

                depo_b[4] = depo_b[3]
                depo_b[3] = depo_b[2]
                depo_b[2] = depo_b[1]
                depo_b[1] = depo_b[0]
                depo_b[0] = ort_b

                # GET THE PIXEL LOCATION AVERAGING PREVIOUS PIXEL VALUES
                result_a = int(((depo_a[0] + depo_a[1] + depo_a[2] + depo_a[3] + depo_a[4]) / 5))
                result_b = int(((depo_b[0] + depo_b[1] + depo_b[2] + depo_b[3] + depo_b[4]) / 5))

                print("ORT")
                # MIDDLE AREA/COORDINATE DETECTION
                frame[result_b - 3:result_b + 4, result_a - 3:result_a + 4] = 0

        else:
            print("Algılancak bir şey yok")

        cv2.imshow("train_gray", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()