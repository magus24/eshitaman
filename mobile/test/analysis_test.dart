import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:eshitaman/core/network/analysis_models.dart';
import 'package:eshitaman/core/network/api_exceptions.dart';
import 'package:eshitaman/core/network/transcription_api_client.dart';

void main() {
  group('AI Analysis Models Serialization Tests', () {
    test('AIAnalysisModel and nested terms serialize and deserialize correctly', () {
      final jsonSample = {
        'recording_id': 'rec_test_ai_001',
        'schema_version': 1,
        'language': 'uz',
        'summary': 'Machine learning va FastAPI tahlili.',
        'key_points': [
          'Model aniqligi 95% bo\'ldi.',
          'FastAPI backendda ishlatildi.'
        ],
        'explanation': 'Ushbu ma\'ruza talabalar uchun muhim.',
        'terms': [
          {
            'term': 'FastAPI',
            'explanation': 'Tezkor Python veb freymvorki',
            'source_segment_ids': ['seg_0001']
          },
          {
            'term': 'Accuracy',
            'explanation': 'Aniqlik ko\'rsatkichi',
            'source_segment_ids': ['seg_0000']
          }
        ],
        'questions': [
          'FastAPI nima uchun tanlandi?',
          'Model aniqligi qancha?'
        ],
        'source_segment_ids': ['seg_0000', 'seg_0001'],
        'provider': 'mock',
        'model': 'mock-model',
        'created_at': '2026-09-26T00:00:00Z',
      };

      final model = AIAnalysisModel.fromJson(jsonSample);
      expect(model.recordingId, 'rec_test_ai_001');
      expect(model.language, 'uz');
      expect(model.summary, 'Machine learning va FastAPI tahlili.');
      expect(model.keyPoints.length, 2);
      expect(model.terms.length, 2);
      expect(model.terms[0].term, 'FastAPI');
      expect(model.terms[0].sourceSegmentIds, ['seg_0001']);
      expect(model.questions.length, 2);

      final serialized = model.toJson();
      expect(serialized['recording_id'], 'rec_test_ai_001');
      expect(serialized['summary'], 'Machine learning va FastAPI tahlili.');
      expect(serialized['terms'], isA<List>());
    });

    test('AnalysisResponseModel serialization', () {
      final jsonSample = {
        'recording_id': 'rec_resp_001',
        'status': 'analyzed',
        'language': 'uz',
        'summary': 'Qisqacha mazmun',
        'key_points_count': 3,
        'terms_count': 2,
        'questions_count': 3,
        'message': 'Tahlil muvaffaqiyatli yakunlandi',
      };

      final model = AnalysisResponseModel.fromJson(jsonSample);
      expect(model.recordingId, 'rec_resp_001');
      expect(model.status, 'analyzed');
      expect(model.keyPointsCount, 3);
      expect(model.termsCount, 2);
      expect(model.questionsCount, 3);
    });
  });

  group('TranscriptionApiClient AI Understanding Tests', () {
    test('triggerAnalyze parses response cleanly', () async {
      final mockClient = MockClient((request) async {
        expect(request.url.path, '/api/v1/recordings/rec_123/analyze');
        expect(request.method, 'POST');
        return http.Response(
          jsonEncode({
            'recording_id': 'rec_123',
            'status': 'analyzed',
            'language': 'uz',
            'summary': 'Tahlil matni',
            'key_points_count': 3,
            'terms_count': 2,
            'questions_count': 2,
            'message': 'Analysis generated successfully',
          }),
          200,
          headers: {'content-type': 'application/json'},
        );
      });

      final apiClient = TranscriptionApiClient(client: mockClient);
      final resp = await apiClient.triggerAnalyze(
        'rec_123',
        contextInstructions: 'Talabalar uchun',
      );

      expect(resp.recordingId, 'rec_123');
      expect(resp.status, 'analyzed');
      expect(resp.keyPointsCount, 3);
      expect(resp.termsCount, 2);
    });

    test('getAnalysis retrieves full AIAnalysisModel', () async {
      final mockClient = MockClient((request) async {
        expect(request.url.path, '/api/v1/recordings/rec_123/analysis');
        expect(request.method, 'GET');
        return http.Response(
          jsonEncode({
            'recording_id': 'rec_123',
            'schema_version': 1,
            'language': 'uz',
            'summary': 'To\'liq tahlil',
            'key_points': ['Fikr 1', 'Fikr 2'],
            'explanation': 'Batafsil tushuntirish',
            'terms': [
              {
                'term': 'AI',
                'explanation': 'Sun\'iy intellekt',
                'source_segment_ids': ['seg_01']
              }
            ],
            'questions': ['Savol 1?'],
            'source_segment_ids': ['seg_01'],
            'provider': 'mock',
            'model': 'mock-model',
            'created_at': '2026-09-26T00:00:00Z',
          }),
          200,
          headers: {'content-type': 'application/json'},
        );
      });

      final apiClient = TranscriptionApiClient(client: mockClient);
      final analysis = await apiClient.getAnalysis('rec_123');

      expect(analysis.recordingId, 'rec_123');
      expect(analysis.summary, 'To\'liq tahlil');
      expect(analysis.keyPoints.length, 2);
      expect(analysis.terms.first.term, 'AI');
    });

    test('triggerAnalyze throws ServerResponseException on 400 error', () async {
      final mockClient = MockClient((request) async {
        return http.Response(
          jsonEncode({
            'detail': {
              'error_code': 'RECORDING_NOT_TRANSCRIBED',
              'message': 'Recording must be transcribed first',
            }
          }),
          400,
          headers: {'content-type': 'application/json'},
        );
      });

      final apiClient = TranscriptionApiClient(client: mockClient);
      expect(
        () => apiClient.triggerAnalyze('rec_untranscribed'),
        throwsA(isA<ServerResponseException>()),
      );
    });
  });
}
