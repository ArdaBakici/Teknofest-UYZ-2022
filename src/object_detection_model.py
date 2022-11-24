import logging
import time

import requests
#from src.yolo import algila
from src.uap import find
from src.yolov7 import analyze
from src.constants import classes, landing_statuses
from src.detected_object import DetectedObject
import cv2
import numpy as np

debug = True # True if you want to see debug messages
do_not_save = False

class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        if do_not_save:
            print("Not sending predictions to evaluation server")
        self.evaulation_server = evaluation_server_url
        # Modelinizi bu kısımda init edebilirsiniz.
        # self.model = get_keras_model() # Örnektir!

    @staticmethod
    def download_image(img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1]  # frame_x.jpg

        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
        bytes_array = np.asarray(bytearray(img_bytes), dtype="uint8")
        img = cv2.imdecode(bytes_array, 1)
        return img

    def process(self, prediction,evaluation_server_url):
        flag = False
        for i in range(5):
            # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
            # Download image (Example)
            img = self.download_image(evaluation_server_url + "media" + prediction.image_url, "./_images/")
            # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
            # ...
            # Nesne tespiti modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
            try:
                frame_results = self.detect(prediction, img)
                flag = False
            except Exception as e:
                print(f"Error at frame: {prediction.image_url} | {e}")
                flag = True
            
            if not flag:
                break
            # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        if flag:
            print(f"Passed frame {prediction.image_url}")
            logging.info(f"Passed frame {prediction.image_url}")
            return "error"
        if do_not_save:
            return "error"
        return frame_results

    def detect(self, prediction, img):
        ### Taşıt - İnsan algılama
        #t1 = time.perf_counter()
        hc_results = analyze(img) # ~0.3
        #t2 = time.perf_counter()
        #print(f'{t2 - t1} seconds to car')
        ### UAP - UAİ Algılma
        #t1 = time.perf_counter()
        uap_results = find(img) # 0.1 very fast
        #t2 = time.perf_counter()
        #print(f'{t2 - t1} seconds to UAP')

        #t1 = time.perf_counter()
        ### İniş uygunluğu algılama
        for i in range(len(uap_results)):
            flag = False
            result = uap_results[i]
            for f in range(1, 5):
                #print(f)
                result[f] = max(0, result[f])
            x0, y0, x1, y1 = result[1], result[2], result[3], result[4]
            for obj in hc_results:
                objx0, objy0, objx1, objy1 = obj[2], obj[3], obj[4], obj[5]
                if (objx0 >= x0 and objx0 <= x1) or (objx1 >= x0 and objx1 <= x1):
                    if (objy0 >= y0 and objy0 <= y1) or (objy1 >= y0 and objy1 <= y1):
                        flag=True
                        break

            if not flag:
                cropped_img = img[y0:y1, x0:x1]
                #print(f"x0 {x0} y0 {y0} x1 {x1} y1 {y1}")
                #print(cropped_img)
                #print(result[0])
                if result[0] == 2:
                    #print("abc")
                    _, binary = cv2.threshold(cropped_img[:,:,0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
                if result[0] == 3:
                    #print("asd")
                    _, binary = cv2.threshold(cropped_img[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                #print(binary)
                closed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                # find all contours in the binary image and check if there is any white pixels bigger than 15 in largest contour
                contours, hierarchy = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(closed_img)
                mask = cv2.fillPoly(mask, pts =[largest_contour], color=(255,255,255))
                n_cls_img = cv2.bitwise_not(closed_img)
                final_mask = cv2.bitwise_and(n_cls_img, mask)
                if np.count_nonzero(final_mask) > 20:
                    uap_results[i].insert(1, "0")
                else:
                    uap_results[i].insert(1, "1")
            else:
                uap_results[i].insert(1, "0")
        #t2 = time.perf_counter()
        #print(f'{t2 - t1} seconds to uygunluk')
        
        # Burada örnek olması amacıyla 20 adet tahmin yapıldığı simüle edilmiştir.
        # Yarışma esnasında modelin tahmin olarak ürettiği sonuçlar kullanılmalıdır.
        # Örneğin :
        # for i in results: # gibi
        for i in uap_results:
            # Modelin tespit ettiği herbir nesne için bir DetectedObject sınıfına ait nesne oluşturularak
            # tahmin modelinin sonuçları parametre olarak verilmelidir.
            d_obj = DetectedObject(i[0],
                                   i[1],
                                   i[2],
                                   i[3],
                                   i[4],
                                   i[5])
            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
            prediction.add_detected_object(d_obj)

            if debug:
                # Modelin tespit ettiği herbir nesne için bir DetectedObject sınıfına ait nesne oluşturularak
                # tahmin modelinin sonuçları parametre olarak verilmelidir.
                # iniş uygun uap siyah, uai cyan, uygun değil uap kırmızı, uai kahverengi
                if i[1] == "0":
                    color = (0, 0, 255) if i[0] == 2 else (0, 255, 255)
                elif i[1] == "1":
                    color = (0, 0, 0) if i[0] == 2 else (255, 255, 0)
                cv2.rectangle(img, (i[2], i[3]), (i[4], i[5]), color, 4)

        for i in hc_results:
            # Modelin tespit ettiği herbir nesne için bir DetectedObject sınıfına ait nesne oluşturularak
            # tahmin modelinin sonuçları parametre olarak verilmelidir.
            for f in range(2, 6):
                if i[f]<0:
                    print("Silent Error")
                i[f] = max(0, i[f])
            d_obj = DetectedObject(i[0],
                                   i[1],
                                   i[2],
                                   i[3],
                                   i[4],
                                   i[5])
            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
            prediction.add_detected_object(d_obj)
            
            if debug:
                color = (255, 0, 0) if i[0] == 0 else (0, 255, 0)
                cv2.rectangle(img, (i[2], i[3]), (i[4], i[5]), color, 4)
        if debug:
            cv2.imwrite(f"./_debug/{prediction.image_url.rsplit('/', 1)[-1]}", img)
            img = cv2.resize(img, (960, 540))
            cv2.imshow("image", img)
            cv2.waitKey(1)
        return prediction
