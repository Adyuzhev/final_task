pipeline {
  agent any
  environment {
        SECRET = credentials('SECRET_GDRIVE_ACC')
  }
  stages {
    stage('setup') {
      steps {
          sh './setup_dvc.sh $SECRET'
      }
    }
    stage('build') {
      steps {
          sh 'docker build -t tag -f Dockerfile .'
      }
    }
    stage('run') {
      steps {
          sh 'docker run -p 8501:8501 tag'
      }
    }
  }
}
