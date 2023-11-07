import easyocr

reader = easyocr.Reader(['en'])
result = reader.readtext('C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\im2.png')

print("These are the results: ", result)