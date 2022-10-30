import boto3

BUCKET_NAME = 'mbti-predict-s3'  
OBJECT_NAME = 'mbti_model.h5' 
FILE_NAME = '/tmp/mbti_model.h5' 

s3 = boto3.client('s3')
s3.download_file(BUCKET_NAME, OBJECT_NAME, FILE_NAME)

mbti_model.load_weights("/tmp/mbti_model.h5")
