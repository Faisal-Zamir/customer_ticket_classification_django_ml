from django.http import JsonResponse
from django.shortcuts import render
from support_analyzer.Model_Files.ticket_classifer import predict_ticket_category

def homepage(request):
    """
    Django view to handle email/ticket classification.
    - On GET: Renders the homepage with the form.
    - On POST (AJAX): Returns JSON response with classification result.
    """
    if request.method == 'POST':
        ticket_text = request.POST.get('ticket_text', '').strip()

        # Validate input
        if not ticket_text:
            return JsonResponse({'success': False, 'error': 'Empty message'})

        # Predict
        ticket_data = {'Document': ticket_text}
        result = predict_ticket_category(ticket_data)

        # Return JSON for AJAX frontend
        return JsonResponse({
            'success': True,
            'predicted_category': result['predicted_category'],
            'confidence': round(result['confidence'], 2)  # Convert to % properly
        })
  
    return render(request, 'support_analyzer/homepage.html')
