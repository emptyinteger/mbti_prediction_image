# aws 에서 제공하는 lambda base image (python)
FROM amazon/aws-lambda-python:3.8

# optional : ensure that pip is up to data
RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

# install git 
RUN yum install git -y

# git clone
RUN git clone https://github.com/emptyinteger/mbti_prediction_image.git 

# install packages
RUN pip install -r mbti_prediction_image/requirements.txt

# git repository 의 lambda_function.py 를 Container 내부의 /var/task/ 로 이동
RUN cp mbti_prediction_image/lambda_function.py /var/task/

# lambda_function.handler 실행
CMD ["lambda_function.handler"]