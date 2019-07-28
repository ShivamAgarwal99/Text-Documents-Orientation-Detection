#end to end code of problem
#input is disoriented PDF
#output oriented PDF

#converting and saving pdf to image dataset
import PyPDF2
import os
from pdf2image import convert_from_path
pdf_dir = r"/home/shivam/Documents/pdf" #input pdf directory
os.chdir(pdf_dir)
for pdf_file in os.listdir(pdf_dir):
        print(pdf_file)
        if pdf_file.endswith(".pdf" or ".PDF"):

            pages = convert_from_path(pdf_file, 100)
            pdf_file = pdf_file[:-4]

            for page in pages:

               page.save("/home/shivam/Documents/pdf/images/%d.jpg" % (pages.index(page)), "JPEG")

#images are running over model
#model predicts the class for every image               
import pandas as pd  
IMAGE_WIDTH=400
IMAGE_HEIGHT=400
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
from keras.preprocessing.image import ImageDataGenerator
             
from tensorflow.keras.models import load_model
model = load_model('/home/shivam/Documents/orientation_detection/Mynewmodel_10.h5')
test_filenames = os.listdir("/home/shivam/Documents/pdf/images")
test_df = pd.DataFrame({
    'filename': test_filenames
})
    
    
nb_samples = test_df.shape[0]

print(nb_samples)
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/home/shivam/Documents/pdf/images", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=32,
    shuffle = False
    
)
predict = model.predict_generator(test_generator)
test_df['category'] = predict.argmax(axis = 1)
test_df['filename'] = [i.split('.', 1)[0] for i in test_df['filename']]
test_df.filename = pd.to_numeric(test_df.filename, errors='coerce')
test_df.sort_values('filename', inplace = True)

#rotating original pdf pages using results of prediction for particular page
pdf_in = open('/home/shivam/Documents/pdf/test1.pdf', 'rb') 
pdf_reader = PyPDF2.PdfFileReader(pdf_in)
pdf_writer = PyPDF2.PdfFileWriter()
for pagenum in range(pdf_reader.numPages):
     page = pdf_reader.getPage(pagenum)
     a= test_df.iloc[pagenum]['category']
     if a == 0:
         page.rotateClockwise(270)
     elif a == 2:
         page.rotateClockwise(90)    
     pdf_writer.addPage(page)
pdf_out = open('/home/shivam/Documents/pdf/rotatednew1.pdf', 'wb')
pdf_writer.write(pdf_out)
pdf_out.close()
pdf_in.close()
