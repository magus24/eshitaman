import 'dart:async';
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:eshitaman/core/network/api_config.dart';
import 'package:eshitaman/core/network/api_exceptions.dart';
import 'package:eshitaman/core/network/audio_upload_service.dart';
import 'package:eshitaman/core/network/upload_response.dart';
import 'package:eshitaman/core/network/transcript_models.dart';
import 'package:eshitaman/core/network/transcription_api_client.dart';
import 'package:eshitaman/features/recording/domain/models/recording_session.dart';


class MockHttpClient extends http.BaseClient {
  final Future<http.StreamedResponse> Function(http.BaseRequest request) handler;
  int requestCount = 0;

  MockHttpClient(this.handler);

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) {
    requestCount++;
    return handler(request);
  }
}

http.StreamedResponse createJsonResponse(int statusCode, Map<String, dynamic> body) {
  final bytes = utf8.encode(jsonEncode(body));
  return http.StreamedResponse(
    Stream.value(bytes),
    statusCode,
    headers: {'content-type': 'application/json'},
  );
}

http.StreamedResponse createRawResponse(int statusCode, String rawBody) {
  final bytes = utf8.encode(rawBody);
  return http.StreamedResponse(
    Stream.value(bytes),
    statusCode,
    headers: {'content-type': 'application/json'},
  );
}

void main() {
  group('ApiConfig Tests', () {
    test('baseUrl resolves cleanly and can be overridden', () {
      ApiConfig.setBaseUrl('https://api.eshitaman.uz');
      expect(ApiConfig.baseUrl, 'https://api.eshitaman.uz');
      expect(ApiConfig.recordingsEndpoint, 'https://api.eshitaman.uz/api/v1/recordings');
      expect(ApiConfig.healthEndpoint, 'https://api.eshitaman.uz/health');

      // Reset
      ApiConfig.setBaseUrl('');
      expect(ApiConfig.baseUrl.isNotEmpty, isTrue);
    });
  });

  group('UploadResponse Model Tests', () {
    test('serialization to and from JSON', () {
      final json = {
        'recording_id': 'rec_999',
        'status': 'uploaded',
        'file_size': 2048,
        'duration': 18.5,
        'mime_type': 'audio/m4a',
        'created_at': '2026-09-26T00:00:00.000Z',
        'message': 'Saved',
      };

      final response = UploadResponse.fromJson(json);
      expect(response.recordingId, 'rec_999');
      expect(response.status, 'uploaded');
      expect(response.fileSize, 2048);
      expect(response.duration, 18.5);
      expect(response.mimeType, 'audio/m4a');
      expect(response.message, 'Saved');

      final serialized = response.toJson();
      expect(serialized['recording_id'], 'rec_999');
      expect(serialized['status'], 'uploaded');
      expect(serialized['file_size'], 2048);
      expect(serialized['duration'], 18.5);
    });
  });

  group('AudioUploadService Tests', () {
    late RecordingSession testSession;
    final dummyBytes = List<int>.generate(256, (i) => i % 256);

    setUp(() {
      testSession = RecordingSession(
        id: 'rec_unit_1',
        filePath: '/mock/storage/audio.m4a',
        duration: const Duration(seconds: 25),
        fileSizeBytes: 256,
        createdAt: DateTime(2026, 9, 26, 10, 0),
        status: RecordingSessionStatus.recorded,
        mimeType: 'audio/m4a',
        codec: 'aac_lc',
      );
    });

    test('successful upload returns parsed UploadResponse', () async {
      final mockClient = MockHttpClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, contains('/api/v1/recordings'));

        return createJsonResponse(201, {
          'recording_id': 'rec_unit_1',
          'status': 'uploaded',
          'file_size': 256,
          'duration': 25.0,
          'mime_type': 'audio/m4a',
          'created_at': '2026-09-26T10:00:00.000Z',
          'message': 'Success',
        });
      });

      final uploadService = AudioUploadService(client: mockClient);
      final response = await uploadService.uploadRecording(
        testSession,
        fileBytesOverride: dummyBytes,
      );

      expect(response.recordingId, 'rec_unit_1');
      expect(response.status, 'uploaded');
      expect(response.fileSize, 256);
      expect(response.duration, 25.0);
      expect(mockClient.requestCount, 1);
    });

    test('client error (400) throws ServerResponseException without retry', () async {
      final mockClient = MockHttpClient((request) async {
        return createJsonResponse(400, {
          'detail': 'Unsupported file extension',
        });
      });

      final uploadService = AudioUploadService(client: mockClient);

      try {
        await uploadService.uploadRecording(
          testSession,
          fileBytesOverride: dummyBytes,
          maxRetries: 3,
          retryDelay: Duration.zero,
        );
        fail('Should have thrown ServerResponseException');
      } on ServerResponseException catch (e) {
        expect(e.statusCode, 400);
        expect(e.message, contains('Unsupported file extension'));
      }

      // Should fail fast on 4xx without retrying
      expect(mockClient.requestCount, 1);
    });

    test('conflict error (409) throws ServerResponseException without retry', () async {
      final mockClient = MockHttpClient((request) async {
        return createJsonResponse(409, {
          'detail': 'Recording with ID rec_unit_1 already exists',
        });
      });

      final uploadService = AudioUploadService(client: mockClient);

      try {
        await uploadService.uploadRecording(
          testSession,
          fileBytesOverride: dummyBytes,
          retryDelay: Duration.zero,
        );
        fail('Should have thrown ServerResponseException');
      } on ServerResponseException catch (e) {
        expect(e.statusCode, 409);
        expect(e.message, contains('already exists'));
      }

      expect(mockClient.requestCount, 1);
    });

    test('server error (500) retries and succeeds on subsequent attempt', () async {
      int attempts = 0;
      final mockClient = MockHttpClient((request) async {
        attempts++;
        if (attempts < 3) {
          return createJsonResponse(500, {'detail': 'Internal Server Error'});
        }
        return createJsonResponse(201, {
          'recording_id': 'rec_unit_1',
          'status': 'uploaded',
          'file_size': 256,
          'duration': 25.0,
          'mime_type': 'audio/m4a',
          'created_at': '2026-09-26T10:00:00.000Z',
          'message': 'Success after retry',
        });
      });

      final uploadService = AudioUploadService(client: mockClient);
      final response = await uploadService.uploadRecording(
        testSession,
        fileBytesOverride: dummyBytes,
        maxRetries: 3,
        retryDelay: Duration.zero,
      );

      expect(response.recordingId, 'rec_unit_1');
      expect(response.status, 'uploaded');
      expect(mockClient.requestCount, 3);
    });

    test('persistent server error (503) fails after maxRetries', () async {
      final mockClient = MockHttpClient((request) async {
        return createJsonResponse(503, {'detail': 'Service Unavailable'});
      });

      final uploadService = AudioUploadService(client: mockClient);

      try {
        await uploadService.uploadRecording(
          testSession,
          fileBytesOverride: dummyBytes,
          maxRetries: 3,
          retryDelay: Duration.zero,
        );
        fail('Should have thrown ServerResponseException');
      } on ServerResponseException catch (e) {
        expect(e.statusCode, 503);
      }

      expect(mockClient.requestCount, 3);
    });

    test('network timeout retries and throws NetworkTimeoutException', () async {
      final mockClient = MockHttpClient((request) async {
        throw TimeoutException('Connection timed out');
      });

      final uploadService = AudioUploadService(client: mockClient);

      try {
        await uploadService.uploadRecording(
          testSession,
          fileBytesOverride: dummyBytes,
          maxRetries: 2,
          retryDelay: Duration.zero,
        );
        fail('Should have thrown NetworkTimeoutException');
      } on NetworkTimeoutException catch (_) {
        // Expected
      }

      expect(mockClient.requestCount, 2);
    });

    test('invalid json response throws InvalidResponseException', () async {
      final mockClient = MockHttpClient((request) async {
        return createRawResponse(200, '<not json at all>');
      });

      final uploadService = AudioUploadService(client: mockClient);

      try {
        await uploadService.uploadRecording(
          testSession,
          fileBytesOverride: dummyBytes,
          retryDelay: Duration.zero,
        );
        fail('Should have thrown InvalidResponseException');
      } on InvalidResponseException catch (_) {
        // Expected
      }
    });

    test('checkHealth returns true on 200 ok', () async {
      final mockClient = MockHttpClient((request) async {
        return createJsonResponse(200, {'status': 'ok'});
      });

      final uploadService = AudioUploadService(client: mockClient);
      final isHealthy = await uploadService.checkHealth();
      expect(isHealthy, isTrue);
    });

    test('checkHealth returns false on failure', () async {
      final mockClient = MockHttpClient((request) async {
        return createJsonResponse(500, {'status': 'error'});
      });

      final uploadService = AudioUploadService(client: mockClient);
      final isHealthy = await uploadService.checkHealth();
      expect(isHealthy, isFalse);
    });
  });

  group('Transcript Models Serialization Tests', () {
    test('TranscriptModel and segments serialization', () {
      final json = {
        'recording_id': 'rec_stt_001',
        'schema_version': 1,
        'language': 'uz',
        'duration_ms': 45000,
        'provider': 'mock',
        'model': 'mock-model',
        'created_at': '2026-09-26T00:00:00Z',
        'text': 'Bugun machine learning haqida gaplashamiz.',
        'segments': [
          {
            'id': 'seg_0000',
            'start_ms': 1000,
            'end_ms': 4500,
            'text': 'Bugun machine learning haqida gaplashamiz.',
            'language': 'uz',
            'confidence': 0.95,
            'chunk_id': 'chunk_0000',
            'words': [
              {
                'word': 'Bugun',
                'start_ms': 1000,
                'end_ms': 1800,
                'confidence': 0.98,
              }
            ],
          }
        ],
      };

      final model = TranscriptModel.fromJson(json);
      expect(model.recordingId, 'rec_stt_001');
      expect(model.language, 'uz');
      expect(model.segments.length, 1);
      expect(model.segments[0].text, 'Bugun machine learning haqida gaplashamiz.');
      expect(model.segments[0].words.length, 1);
      expect(model.segments[0].words[0].word, 'Bugun');

      final serialized = model.toJson();
      expect(serialized['recording_id'], 'rec_stt_001');
      expect(serialized['segments'][0]['words'][0]['word'], 'Bugun');
    });
  });

  group('TranscriptionApiClient Tests', () {
    test('triggerTranscribe parses response cleanly', () async {
      final mockClient = MockHttpClient((request) async {
        expect(request.url.path.endsWith('/transcribe'), isTrue);
        return createJsonResponse(200, {
          'recording_id': 'rec_trans_1',
          'status': 'transcribed',
          'language': 'uz',
          'segments_count': 5,
          'duration_ms': 60000,
          'message': 'Transkripsiya yakunlandi',
        });
      });

      final client = TranscriptionApiClient(client: mockClient);
      final response = await client.triggerTranscribe('rec_trans_1');
      expect(response.recordingId, 'rec_trans_1');
      expect(response.status, 'transcribed');
      expect(response.segmentsCount, 5);
      expect(response.language, 'uz');
    });

    test('getTranscript retrieves full transcript', () async {
      final mockClient = MockHttpClient((request) async {
        expect(request.url.path.endsWith('/transcript'), isTrue);
        return createJsonResponse(200, {
          'recording_id': 'rec_trans_1',
          'schema_version': 1,
          'language': 'uz',
          'duration_ms': 30000,
          'provider': 'mock',
          'model': 'mock-model',
          'created_at': '2026-09-26T00:00:00Z',
          'text': 'Bugun dars boshlandi.',
          'segments': [],
        });
      });

      final client = TranscriptionApiClient(client: mockClient);
      final transcript = await client.getTranscript('rec_trans_1');
      expect(transcript.recordingId, 'rec_trans_1');
      expect(transcript.text, 'Bugun dars boshlandi.');
    });

    test('404 on transcript throws ServerResponseException', () async {
      final mockClient = MockHttpClient((request) async {
        return createJsonResponse(404, {'detail': 'Transcript not found'});
      });

      final client = TranscriptionApiClient(client: mockClient);
      expect(
        () => client.getTranscript('rec_non_existent'),
        throwsA(isA<ServerResponseException>()),
      );
    });
  });
}

