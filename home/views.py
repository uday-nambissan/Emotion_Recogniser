from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import librosa
import soundfile
import pickle
import numpy as np

def extract_features(file_name, mfcc, chroma,mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result= np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result=np.hstack( (result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result=np.hstack((result, mel))
    return result 



def home(request):
    context={}
    if request.method=="POST":
        uploaded_file=request.FILES['audio']
        fs=FileSystemStorage()
        name=fs.save(uploaded_file.name, uploaded_file)
        context['url']=fs.url(name)

        loaded_model=pickle.load(open('.\static\modelForPrediction1','rb'))

        feature=extract_features('.\media\\'+name, mfcc=True, chroma=True, mel=True)
        feature=feature.reshape(1,-1)
        prediction=loaded_model.predict(feature)
        context['result']=prediction[0]



    return render(request, "index.html",context)


