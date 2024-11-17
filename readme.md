4v0 : cpu normal 
4v1 : gpu included 
4v2 : gpu included with resize

The project and idea initially started as my friend Nayeem's undergrad thesis, focusing on vehicle detection and counting for CNGs and rickshaws in Bangladesh using deep learning techniques. Nayeem trained the system with YOLOv7, utilizing various photos and videos, and successfully defended his thesis. Later, I enhanced the model by experimenting with YOLOv8 and YOLOv11, optimizing its accuracy and performance. We integrated machine learning-based features like tracking vehicle centroids across frames to detect when vehicles cross a designated line. The system aims to enhance traffic management and data collection for transportation in Bangladesh.

The project use convolutional neural networks (CNNs) through YOLO (You Only Look Once) architecture for real-time object detection. YOLOv7, YOLOv8, and YOLOv11 models were fine-tuned for the specific task of detecting and counting CNGs and rickshaws. The deep learning model was trained on annotated datasets consisting of diverse images and videos, employing techniques like transfer learning for improved accuracy. To enhance vehicle tracking, the system uses centroid tracking across frames, tracking algorithms to monitor vehicles as they cross designated lines, ensuring  vehicle flow analysis.
