# aerosol-website
Worldwide prediction of n100 concentrations based on carbon monoxide and temperature reanalysis data


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
