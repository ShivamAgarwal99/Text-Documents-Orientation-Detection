#Code for converting pdf to images
import os
from pdf2image import convert_from_path
pdf_dir = r"/home/shivam/Downloads/AMI_NEW_DUMP/AMI_CITY_OF_BREWTON"
os.chdir(pdf_dir)
for pdf_file in os.listdir(pdf_dir):
        print(pdf_file)
        print('this')
        if pdf_file.endswith(".pdf" or ".PDF"):

            pages = convert_from_path(pdf_file, 100)
            pdf_file = pdf_file[:-4]

            for page in pages:

               page.save("/home/shivam/Documents/orientation_detection/temp/%s-page%d.jpg" % (pdf_file,pages.index(page)), "JPEG")
