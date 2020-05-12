FROM python:3.8-slim

COPY ./webapp/requirements.txt /tmp/requirements.txt

RUN pip3 install --upgrade -r /tmp/requirements.txt

RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
headless = true\n\
enableCORS=false\n\
" > /root/.streamlit/config.toml'

COPY ./webapp /opt/webapp/
WORKDIR /opt/webapp

EXPOSE $PORT
CMD streamlit run --server.port $PORT app.py