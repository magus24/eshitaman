import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:eshitaman/core/network/api_exceptions.dart';
import 'package:eshitaman/core/network/groq_client.dart';
import 'package:eshitaman/core/network/groq_config.dart';
import 'package:eshitaman/features/processing/data/on_device_pipeline.dart';

const testKey = 'gsk_test_key';

class RecordingMockClient extends http.BaseClient {
  final Future<http.StreamedResponse> Function(http.BaseRequest request) handler;
  final List<http.BaseRequest> requests = [];

  RecordingMockClient(this.handler);

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) {
    requests.add(request);
    return handler(request);
  }
}

http.StreamedResponse jsonResponse(int statusCode, Map<String, dynamic> body) {
  final bytes = utf8.encode(jsonEncode(body));
  return http.StreamedResponse(
    Stream.value(bytes),
    statusCode,
    headers: {'content-type': 'application/json'},
  );
}

void main() {
  group('GroqConfig Tests', () {
    test('exposes Groq endpoints and limits', () {
      expect(GroqConfig.baseUrl, 'https://api.groq.com/openai/v1');
      expect(GroqConfig.chatEndpoint, endsWith('/chat/completions'));
      expect(GroqConfig.transcribeEndpoint, endsWith('/audio/transcriptions'));
      expect(GroqConfig.maxUploadBytes, 25 * 1024 * 1024);
    });

    test('missing key produces an actionable message', () {
      expect(GroqConfig.misconfigurationReasonFor(''), isNotNull);
      expect(GroqConfig.misconfigurationReasonFor('   '), isNotNull);
      expect(GroqConfig.misconfigurationReasonFor(testKey), isNull);
    });
  });

  group('GroqClient transcription', () {
    test('parses verbose_json segments with timestamps', () async {
      final mock = RecordingMockClient((_) async => jsonResponse(200, {
            'text': 'Salom, bu dars testi.',
            'language': 'uz',
            'duration': 12.5,
            'segments': [
              {'start': 0.0, 'end': 5.5, 'text': 'Salom,'},
              {'start': 5.5, 'end': 12.5, 'text': 'bu dars testi.'},
            ],
          }));

      final client = GroqClient(client: mock, apiKey: testKey);
      final result = await client.transcribe(
        audioBytes: List<int>.filled(128, 1),
        filename: 'audio.m4a',
        mimeType: 'audio/m4a',
      );

      expect(result.text, 'Salom, bu dars testi.');
      expect(result.language, 'uz');
      expect(result.durationSeconds, 12.5);
      expect(result.segments.length, 2);
      expect(result.segments.first.startMs, 0);
      expect(result.segments.first.endMs, 5500);
      expect(result.segments.last.id, 'seg_1');

      final request = mock.requests.single as http.MultipartRequest;
      expect(request.url.path, endsWith('/audio/transcriptions'));
      expect(request.headers['Authorization'], 'Bearer $testKey');
      expect(request.fields['model'], GroqConfig.sttModel);
      expect(request.fields['response_format'], 'verbose_json');
      expect(request.fields['timestamp_granularities[]'], 'segment');
      expect(request.files.single.field, 'file');
    });

    test('falls back to plain json when granularities are rejected', () async {
      var call = 0;
      final mock = RecordingMockClient((_) async {
        call++;
        if (call == 1) {
          return jsonResponse(400, {
            'error': {'message': 'granularities not supported'}
          });
        }
        return jsonResponse(200, {'text': 'Oddiy matn'});
      });

      final client = GroqClient(client: mock, apiKey: testKey);
      final result = await client.transcribe(
        audioBytes: List<int>.filled(64, 1),
        filename: 'audio.m4a',
      );

      expect(call, 2);
      expect(result.text, 'Oddiy matn');
      expect(result.segments, isEmpty);
      final fallback = mock.requests.last as http.MultipartRequest;
      expect(fallback.fields['response_format'], 'json');
      expect(fallback.fields.containsKey('timestamp_granularities[]'), isFalse);
    });

    test('rejects empty and oversized payloads before any request', () async {
      var calls = 0;
      final mock = RecordingMockClient((_) async {
        calls++;
        return jsonResponse(200, {'text': 'x'});
      });
      final client = GroqClient(client: mock, apiKey: testKey);

      await expectLater(
        client.transcribe(audioBytes: const [], filename: 'a.m4a'),
        throwsA(isA<ApiException>()),
      );
      await expectLater(
        client.transcribe(
          audioBytes: List<int>.filled(GroqConfig.maxUploadBytes + 1, 0),
          filename: 'a.m4a',
        ),
        throwsA(isA<ApiException>()),
      );
      expect(calls, 0);
    });

    test('missing api key fails fast', () async {
      final mock = RecordingMockClient((_) async => jsonResponse(200, {}));
      final client = GroqClient(client: mock, apiKey: '');

      await expectLater(
        client.transcribe(audioBytes: [1, 2, 3], filename: 'a.m4a'),
        throwsA(isA<ApiException>()),
      );
      expect(mock.requests, isEmpty);
    });
  });

  group('GroqClient chat', () {
    test('requests json_object mode and parses the object', () async {
      final mock = RecordingMockClient((_) async => jsonResponse(
            200,
            {
              'choices': [
                {
                  'message': {
                    'content': '{"summary":"Xulosa","key_points":["a","b","c"]}'
                  }
                }
              ]
            },
          ));

      final client = GroqClient(client: mock, apiKey: testKey);
      final json = await client.completeJson(
        systemPrompt: 'system',
        userPrompt: 'user',
      );

      expect(json['summary'], 'Xulosa');
      expect(json['key_points'], ['a', 'b', 'c']);

      final request = mock.requests.single;
      expect(request.url.path, endsWith('/chat/completions'));
      final sent = jsonDecode((request as http.Request).body) as Map<String, dynamic>;
      expect(sent['model'], GroqConfig.chatModel);
      expect(sent['response_format'], {'type': 'json_object'});
      expect((sent['messages'] as List).first['role'], 'system');
    });

    test('strips markdown fences around the JSON payload', () async {
      final mock = RecordingMockClient((_) async => jsonResponse(200, {
            'choices': [
              {
                'message': {
                  'content': '```json\n{"summary":"Fence bilan"}\n```'
                }
              }
            ]
          }));

      final client = GroqClient(client: mock, apiKey: testKey);
      final json = await client.completeJson(
        systemPrompt: 's',
        userPrompt: 'u',
      );
      expect(json['summary'], 'Fence bilan');
    });

    test('falls back to reasoning when content is empty', () async {
      final mock = RecordingMockClient((_) async => jsonResponse(200, {
            'choices': [
              {
                'message': {
                  'content': '',
                  'reasoning': '{"summary":"reasoning ichidan"}'
                }
              }
            ]
          }));

      final client = GroqClient(client: mock, apiKey: testKey);
      final json = await client.completeJson(
        systemPrompt: 's',
        userPrompt: 'u',
      );
      expect(json['summary'], 'reasoning ichidan');
    });

    test('throws when no JSON object can be recovered', () async {
      final mock = RecordingMockClient((_) async =>
          jsonResponse(200, {'choices': [{'message': {'content': 'yo‘q'}}]}));

      final client = GroqClient(client: mock, apiKey: testKey);
      await expectLater(
        client.completeJson(systemPrompt: 's', userPrompt: 'u'),
        throwsA(isA<InvalidResponseException>()),
      );
    });

    test('maps Groq status codes to readable Uzbek errors', () async {
      final cases = {
        401: 'API kaliti noto‘g‘ri',
        413: 'juda katta',
        429: 'limitiga',
        500: 'vaqtincha band',
      };

      for (final entry in cases.entries) {
        final mock = RecordingMockClient((_) async => jsonResponse(
              entry.key,
              {'error': {'message': 'raw'}},
            ));
        final client = GroqClient(client: mock, apiKey: testKey);

        await expectLater(
          client.complete(systemPrompt: 's', userPrompt: 'u'),
          throwsA(
            isA<ServerResponseException>().having(
              (e) => e.message,
              'message',
              contains(entry.value),
            ),
          ),
        );
      }
    });
  });

  group('OnDeviceAnalysisPipeline', () {
    test('maps the LLM JSON payload onto AIAnalysisModel', () async {
      final mock = RecordingMockClient((_) async => jsonResponse(200, {
            'choices': [
              {
                'message': {
                  'content': jsonEncode({
                    'language': 'Uzbek',
                    'summary': 'Neyron tarmoqlar haqida dars.',
                    'key_points': ['birinchi', 'ikkinchi', 'uchinchi', '  '],
                    'explanation': 'Batafsil tushuntirish.',
                    'terms': [
                      {'term': 'CNN', 'explanation': 'Tasvir modeli'},
                      {'term': '', 'explanation': 'bo‘sh'},
                    ],
                    'questions': ['Savol 1?', 'Savol 2?', 'Savol 3?'],
                  })
                }
              }
            ]
          }));

      final pipeline = OnDeviceAnalysisPipeline(
        client: GroqClient(client: mock, apiKey: testKey),
      );
      final analysis = await pipeline.analyzeTranscript(
        recordingId: 'rec_1',
        transcriptText: 'Dars matni.',
      );

      expect(analysis.recordingId, 'rec_1');
      expect(analysis.language, 'uz');
      expect(analysis.provider, 'groq');
      expect(analysis.model, GroqConfig.chatModel);
      expect(analysis.summary, 'Neyron tarmoqlar haqida dars.');
      expect(analysis.keyPoints.length, 3);
      expect(analysis.terms.length, 1);
      expect(analysis.terms.first.term, 'CNN');
      expect(analysis.questions.length, 3);
    });

    test('sends the transcript as grounding context for chat', () async {
      final mock = RecordingMockClient((_) async => jsonResponse(200, {
            'choices': [
              {
                'message': {
                  'content': 'Asosiy g\'oya — neyron tarmoqlar.'
                }
              }
            ]
          }));

      final pipeline = OnDeviceAnalysisPipeline(
        client: GroqClient(client: mock, apiKey: testKey),
      );
      final answer = await pipeline.askAboutTranscript(
        transcriptText: 'Bugungi darsda neyron tarmoqlar.',
        conversationHistory: 'Savol: avvalgi savol?',
        question: 'Asosiy g\'oyasi nima?',
      );

      expect(answer, contains('neyron'));

      final sent = jsonDecode((mock.requests.single as http.Request).body)
          as Map<String, dynamic>;
      final userPrompt =
          (sent['messages'] as List).last['content'] as String;
      expect(userPrompt, contains('Bugungi darsda neyron tarmoqlar.'));
      expect(userPrompt, contains('avvalgi savol?'));
      expect(userPrompt, contains('Asosiy g\'oyasi nima?'));
    });

    test('truncates very long transcripts before sending them', () async {
      final mock = RecordingMockClient((_) async => jsonResponse(200, {
            'choices': [
              {
                'message': {
                  'content': 'Javob'
                }
              }
            ]
          }));

      final pipeline = OnDeviceAnalysisPipeline(
        client: GroqClient(client: mock, apiKey: testKey),
      );
      await pipeline.askAboutTranscript(
        transcriptText: 'x' * (GroqConfig.transcriptCharLimit + 5000),
        conversationHistory: '',
        question: 'Savol?',
      );

      final sent = jsonDecode((mock.requests.single as http.Request).body)
          as Map<String, dynamic>;
      final userPrompt =
          (sent['messages'] as List).last['content'] as String;
      expect(userPrompt, contains('[Matn qisqartirildi]'));
      expect(userPrompt.length, lessThan(GroqConfig.transcriptCharLimit + 500));
    });
  });
}
