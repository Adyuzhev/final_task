pipeline {
  agent any
  stages {
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
