from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout 
from django.contrib.auth.decorators import login_required
from .forms import CreateUserForm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


@login_required(login_url='login')
def home(request):
    return render(request, 'index.html')

@login_required(login_url='login')
def predict(request):
    return render (request, 'predict.html')


def predictions(request):
    data = pd.read_csv('Dataset\Housing.csv')
    data = data.drop([ 'prefarea'], axis = 1)
    data.dropna(inplace=True)


    X = data.drop(["price"], axis=1)
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



    numeric_features = ["area", "bedrooms", "bathrooms", "stories", "parking",]
    categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating' , 'airconditioning','furnishingstatus']


    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
    ])
    
    
    model.fit(X_train, y_train)
    

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    obj5 = str(request.GET['n5'])
    obj6 = str(request.GET['n6'])
    obj7 = str(request.GET['n7'])
    obj8 = str(request.GET['n8'])
    obj9 = str(request.GET['n9'])
    var10 = float(request.GET['n10'])
    var11 = str(request.GET['n11'])
    

    input_data = pd.DataFrame({
        'area': [var1],
        'bedrooms': [var2],
        'bathrooms': [var3],
        'stories': [var4],
        'mainroad': [obj5],
        'guestroom': [obj6],
        'basement': [obj7],
        'hotwaterheating': [obj8],
        'airconditioning': [obj9],
        'parking': [var10],
        'furnishingstatus': [var11],
    })


    predictions = model.predict(input_data)
    predictions = round(predictions[0])

    

    price = 'The predicted price is: $'+str(predictions)

    return render (request, 'result.html', {'result2':price})

def about(request):
    return render(request, 'about.html')

def signup(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        form = CreateUserForm()

        if request.method =='POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for '+ user)
                return redirect('login')
            

    context = {'form':form}
    return render(request, 'signup.html', context)

def loginpage(request):
    if request.method == 'POST':
        username=request.POST.get('username')
        password=request.POST.get('password')
    
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.info(request,"Username or password incorrect")
            
    context = {}
    return render(request, 'login.html', context)

def logoutUser(request):
    logout(request)
    return redirect('login')
    
# Create your views here.
