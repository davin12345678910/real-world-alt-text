from ultralytics import YOLO

model = YOLO('yolov8n.pt')

print("I AM GOING TO BE CALLING YOLOV8")
results = model(
    ["C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\test-image\\a_group_of_people_biking_towards.jpg"])
print("WE FINISHED CALLING YOLOV8")