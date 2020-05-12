FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    libproj-dev \
    proj-data \
    proj-bin  \
    libgeos-dev \
    gcc \
    g++ \
    git \
    python3-eccodes

RUN pip3 install --upgrade cython numpy

# python: LineString.cpp:125: const geos::geom::CoordinateSequence* 
# geos::geom::LineString::getCoordinatesRO() 
# const: Assertion `nullptr != points.get()' failed.
# Hotfix: 
# shapely should be built from source instead of using the binary
RUN pip3 install --upgrade shapely --no-binary shapely

# 14.3.2020: cartopy 0.17.0 depends on proj4 whereas I get 5
# Fixed on GitHub, but not released yet 
# => additional deps: gcc, g++ and git 
RUN pip3 install git+https://github.com/SciTools/cartopy.git --no-binary cartopy

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