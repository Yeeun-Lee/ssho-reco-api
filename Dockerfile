FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:fkrull/deadsnakes
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.6 python3.6-dev python3-pip python3-setuptools python3-wheel gcc
RUN apt-get install -y git
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y libgl1-mesa-glx

COPY . /app
WORKDIR /app

# pip 업그레이드
RUN python3.6 -m pip install pip --upgrade

# 작업 디렉토리에 있는 requirements.txt로 패키지 설치
RUN pip3 install -r requirements.txt

# 컨테이너에서 실행될 명령어. 컨테이거나 실행되면 app.py를 실행시킨다.
CMD python3.6 app.py