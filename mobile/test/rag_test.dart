import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:eshitaman/core/network/rag_models.dart';
import 'package:eshitaman/core/network/transcription_api_client.dart';
import 'package:eshitaman/features/chat/presentation/screens/ai_chat_screen.dart';

void main() {
  group('RAG Models Serialization Tests', () {
    test('RAGSourceModel serialization & formatting', () {
      final json = {
        'chunk_id': 'chunk_001',
        'start_time': 65.5,
        'end_time': 125.0,
        'text': 'Gradient descent optimallash algoritmi',
        'relevance': 0.89,
      };

      final model = RAGSourceModel.fromJson(json);
      expect(model.chunkId, 'chunk_001');
      expect(model.startTime, 65.5);
      expect(model.endTime, 125.0);
      expect(model.text, 'Gradient descent optimallash algoritmi');
      expect(model.relevance, 0.89);
      expect(model.formattedTimeRange, '01:05 – 02:05');

      final serialized = model.toJson();
      expect(serialized['chunk_id'], 'chunk_001');
      expect(serialized['relevance'], 0.89);
    });

    test('RAGChatResponseModel serialization', () {
      final json = {
        'recording_id': 'rec_rag_123',
        'answer': 'Gradient descent yo‘qotish funksiyasini minimallashtiradi.',
        'sources': [
          {
            'chunk_id': 'c1',
            'start_time': 10.0,
            'end_time': 30.0,
            'text': 'Yo‘qotish funksiyasi haqida gap',
            'relevance': 0.95,
          }
        ],
        'language': 'uz',
        'message': 'Success',
      };

      final model = RAGChatResponseModel.fromJson(json);
      expect(model.recordingId, 'rec_rag_123');
      expect(model.answer, contains('minimallashtiradi'));
      expect(model.sources.length, 1);
      expect(model.sources.first.chunkId, 'c1');
      expect(model.language, 'uz');
    });

    test('RAGIndexResponseModel serialization', () {
      final json = {
        'recording_id': 'rec_rag_123',
        'status': 'indexed',
        'chunks_count': 5,
        'embedding_dimension': 1536,
        'message': 'Transcript indexed successfully',
      };

      final model = RAGIndexResponseModel.fromJson(json);
      expect(model.recordingId, 'rec_rag_123');
      expect(model.status, 'indexed');
      expect(model.chunksCount, 5);
      expect(model.embeddingDimension, 1536);
    });
  });

  group('TranscriptionApiClient RAG Tests', () {
    test('triggerIndex parses response cleanly', () async {
      final mockClient = MockClient((request) async {
        expect(request.url.path, '/api/v1/recordings/rec_001/index');
        return http.Response(
          jsonEncode({
            'recording_id': 'rec_001',
            'status': 'indexed',
            'chunks_count': 4,
            'embedding_dimension': 1536,
            'message': 'Indexed',
          }),
          200,
        );
      });

      final apiClient = TranscriptionApiClient(client: mockClient);
      final res = await apiClient.triggerIndex('rec_001');
      expect(res.recordingId, 'rec_001');
      expect(res.status, 'indexed');
      expect(res.chunksCount, 4);
    });

    test('askQuestion sends request and parses grounded answer and sources', () async {
      final mockClient = MockClient((request) async {
        expect(request.url.path, '/api/v1/recordings/rec_001/chat');
        final body = jsonDecode(request.body) as Map<String, dynamic>;
        expect(body['question'], 'Gradient descent nima?');

        return http.Response(
          jsonEncode({
            'recording_id': 'rec_001',
            'answer': 'Gradient descent optimallash algoritmi hisoblanadi.',
            'sources': [
              {
                'chunk_id': 'chunk_1',
                'start_time': 12.0,
                'end_time': 28.0,
                'text': 'Gradient descent bu asosiy algoritm...',
                'relevance': 0.92,
              }
            ],
            'language': 'uz',
            'message': 'Answered',
          }),
          200,
        );
      });

      final apiClient = TranscriptionApiClient(client: mockClient);
      final res = await apiClient.askQuestion('rec_001', 'Gradient descent nima?');
      expect(res.recordingId, 'rec_001');
      expect(res.answer, contains('optimallash algoritmi'));
      expect(res.sources.length, 1);
      expect(res.sources.first.formattedTimeRange, '00:12 – 00:28');
    });
  });

  group('AiChatScreen Widget Tests', () {
    testWidgets('renders suggested questions and handles chat interaction', (tester) async {
      final mockClient = MockClient((request) async {
        return http.Response(
          jsonEncode({
            'recording_id': 'rec_widget_01',
            'answer': 'Model accuracy 95% ga yetdi.',
            'sources': [
              {
                'chunk_id': 'src_1',
                'start_time': 5.0,
                'end_time': 15.0,
                'text': 'Model accuracy 95 percent',
                'relevance': 0.91,
              }
            ],
            'language': 'uz',
            'message': 'OK',
          }),
          200,
        );
      });

      final apiClient = TranscriptionApiClient(client: mockClient);

      await tester.pumpWidget(
        MaterialApp(
          home: AiChatScreen(
            recordingTitle: 'AI Darsi #1',
            recordingId: 'rec_widget_01',
            initialQuestions: const ['Model natijasi qanday?'],
            apiClient: apiClient,
          ),
        ),
      );

      // Verify header and initial message
      expect(find.text('AI Darsi #1'), findsOneWidget);
      expect(find.text('Model natijasi qanday?'), findsOneWidget);

      // Tap suggested question
      await tester.tap(find.text('Model natijasi qanday?'));
      await tester.pump(); // Start request

      // Let async response complete
      await tester.pump(const Duration(milliseconds: 100));
      await tester.pumpAndSettle();

      // Check user message and AI response appear
      expect(find.text('Model natijasi qanday?'), findsNWidgets(2)); // Chip and message bubble
      expect(find.text('Model accuracy 95% ga yetdi.'), findsOneWidget);

      // Check source citation time range appears
      expect(find.text('00:05 – 00:15'), findsOneWidget);
    });
  });
}
