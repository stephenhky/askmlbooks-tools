FROM python:3.11

ADD . /code

WORKDIR /code

RUN apt-get update && \
  apt-get install -y \
  gcc \
  g++ \
  libc6 \
  gfortran \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev

RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install awslambdaric boto3

ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD [ "lambda_function.lambda_handler" ]
