# views.py
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response


# --- List View (GET /settings/) ---
@api_view(['GET'])
def health(request):
    try:
        return Response({'message': 'Service is healthy'}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': f'An unexpected error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)