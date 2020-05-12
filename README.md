# aerosol-website
Worldwide prediction of n100 concentrations based on carbon monoxide and temperature reanalysis data

### Live at

https://aerosol.herokuapp.com/

### Docker :whale:

```bash
$ docker build -t aerosol-app .
$ docker run -it -p 8080:8080 aerosol-app
```

### GCP :cloud:

```bash
$ gcloud projects list
$ gcloud config set project <project-name>
$ gcloud app deploy
$ gcloud app browse
```

### Heroku Container Registry

```bash
$ heroku login
$ heroku container:login
$ heroku container:push web
$ heroku container:release web
```

### Howto

https://docs.streamlit.io/