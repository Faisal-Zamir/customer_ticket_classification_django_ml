from django.shortcuts import render

def homepage(request):
    return render(request, 'support_analyzer/homepage.html')