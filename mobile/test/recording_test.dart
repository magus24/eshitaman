import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:record/record.dart';
import 'package:eshitaman/features/recording/domain/models/recording_session.dart';
import 'package:eshitaman/features/recording/data/repositories/recording_repository.dart';
import 'package:eshitaman/features/recording/data/services/audio_recording_service.dart';

class FakeAudioRecorder extends Fake implements AudioRecorder {
  bool permission = true;
  bool isStarted = false;
  bool pausedState = false;
  String? lastPath;
  RecordConfig? lastConfig;
  int startCallCount = 0;
  int pauseCallCount = 0;
  int resumeCallCount = 0;
  int stopCallCount = 0;
  int cancelCallCount = 0;

  @override
  Future<bool> hasPermission({bool request = true}) async => permission;

  @override
  Future<bool> isPaused() async => pausedState;

  @override
  Future<bool> isRecording() async => isStarted && !pausedState;

  @override
  Future<void> start(RecordConfig config, {required String path}) async {
    startCallCount++;
    lastConfig = config;
    lastPath = path;
    isStarted = true;
    pausedState = false;
  }

  @override
  Future<void> pause() async {
    pauseCallCount++;
    pausedState = true;
  }

  @override
  Future<void> resume() async {
    resumeCallCount++;
    pausedState = false;
  }

  @override
  Future<String?> stop() async {
    stopCallCount++;
    isStarted = false;
    pausedState = false;
    return lastPath;
  }

  @override
  Future<void> cancel() async {
    cancelCallCount++;
    isStarted = false;
    pausedState = false;
  }

  @override
  Future<Amplitude> getAmplitude() async =>
      Amplitude(current: -15.0, max: -2.0);

  @override
  Future<void> dispose() async {}
}

void main() {
  group('RecordingSession Model Tests', () {
    test('should correctly serialize to and from JSON with full metadata', () {
      final session = RecordingSession(
        id: 'rec_12345',
        filePath: '/storage/recordings/rec_12345/audio.m4a',
        duration: const Duration(minutes: 1, seconds: 35),
        fileSizeBytes: 1048576,
        createdAt: DateTime(2026, 9, 25, 22, 0, 0),
        status: RecordingSessionStatus.recorded,
        mimeType: 'audio/m4a',
        codec: 'aac_lc',
        sampleRate: 44100,
        bitRate: 128000,
        channelCount: 1,
      );

      expect(session.durationMillis, 95000);
      expect(session.fileSize, 1048576);

      final json = session.toJson();
      expect(json['id'], 'rec_12345');
      expect(json['durationSeconds'], 95);
      expect(json['durationMillis'], 95000);
      expect(json['fileSize'], 1048576);
      expect(json['fileSizeBytes'], 1048576);
      expect(json['status'], 'recorded');
      expect(json['mimeType'], 'audio/m4a');
      expect(json['codec'], 'aac_lc');
      expect(json['sampleRate'], 44100);
      expect(json['bitRate'], 128000);
      expect(json['channelCount'], 1);

      final deserialized = RecordingSession.fromJson(json);
      expect(deserialized.id, session.id);
      expect(deserialized.filePath, session.filePath);
      expect(deserialized.duration, session.duration);
      expect(deserialized.durationMillis, 95000);
      expect(deserialized.fileSizeBytes, session.fileSizeBytes);
      expect(deserialized.fileSize, session.fileSize);
      expect(deserialized.status, RecordingSessionStatus.recorded);
      expect(deserialized.mimeType, 'audio/m4a');
      expect(deserialized.codec, 'aac_lc');
      expect(deserialized.sampleRate, 44100);
      expect(deserialized.bitRate, 128000);
      expect(deserialized.channelCount, 1);
    });

    test('copyWith updates fields correctly without mutating originals', () {
      final session = RecordingSession(
        id: 'rec_1',
        filePath: '/audio.m4a',
        duration: const Duration(seconds: 10),
        fileSizeBytes: 500,
        createdAt: DateTime(2026, 9, 25, 12, 0),
        sampleRate: 44100,
      );

      final updated = session.copyWith(
        status: RecordingSessionStatus.failed,
        errorMessage: 'Microphone busy',
        bitRate: 128000,
      );

      expect(updated.id, 'rec_1');
      expect(updated.status, RecordingSessionStatus.failed);
      expect(updated.errorMessage, 'Microphone busy');
      expect(updated.sampleRate, 44100);
      expect(updated.bitRate, 128000);
      expect(session.status, RecordingSessionStatus.recorded);
    });
  });

  group('RecordingRepository Tests', () {
    late RecordingRepository repository;

    setUp(() {
      repository = RecordingRepository();
      repository.clear();
    });

    test('addSession adds session to top and notifies listeners', () {
      bool notified = false;
      repository.addListener(() {
        notified = true;
      });

      final session = RecordingSession(
        id: 'rec_test',
        filePath: '/test.m4a',
        duration: const Duration(seconds: 30),
        fileSizeBytes: 12000,
        createdAt: DateTime.now(),
      );

      repository.addSession(session);

      expect(repository.sessions.length, 1);
      expect(repository.sessions.first.id, 'rec_test');
      expect(notified, isTrue);
    });

    test('removeSession removes matching session', () {
      final session1 = RecordingSession(
        id: 'rec_1',
        filePath: '/1.m4a',
        duration: const Duration(seconds: 10),
        fileSizeBytes: 100,
        createdAt: DateTime.now(),
      );
      final session2 = RecordingSession(
        id: 'rec_2',
        filePath: '/2.m4a',
        duration: const Duration(seconds: 20),
        fileSizeBytes: 200,
        createdAt: DateTime.now(),
      );

      repository.addSession(session1);
      repository.addSession(session2);
      expect(repository.sessions.length, 2);

      repository.removeSession('rec_1');
      expect(repository.sessions.length, 1);
      expect(repository.sessions.first.id, 'rec_2');
    });
  });

  group('Duration & Pause Calculation Logic Tests', () {
    test('paused time is excluded from total accumulated duration', () async {
      Duration accumulatedDuration = Duration.zero;
      final stopwatch = Stopwatch();

      // Segment 1: Record 60ms
      stopwatch.start();
      await Future.delayed(const Duration(milliseconds: 60));
      stopwatch.stop();
      accumulatedDuration += stopwatch.elapsed;
      stopwatch.reset();

      // Pause interval: 90ms (Stopwatch inactive)
      await Future.delayed(const Duration(milliseconds: 90));

      // Segment 2: Resume record 60ms
      stopwatch.start();
      await Future.delayed(const Duration(milliseconds: 60));
      stopwatch.stop();
      accumulatedDuration += stopwatch.elapsed;

      // Total recorded elapsed duration should reflect only active recording time (~120ms)
      expect(accumulatedDuration.inMilliseconds, greaterThanOrEqualTo(100));
      expect(accumulatedDuration.inMilliseconds, lessThan(200));
    });

    test('multiple pause and resume cycles accumulate accurately', () async {
      Duration accumulatedDuration = Duration.zero;
      final stopwatch = Stopwatch();

      for (int i = 0; i < 3; i++) {
        stopwatch.start();
        await Future.delayed(const Duration(milliseconds: 30));
        stopwatch.stop();
        accumulatedDuration += stopwatch.elapsed;
        stopwatch.reset();

        // Pause
        await Future.delayed(const Duration(milliseconds: 40));
      }

      // 3 active segments of 30ms = ~90ms active, pause interval of 120ms excluded
      expect(accumulatedDuration.inMilliseconds, greaterThanOrEqualTo(75));
      expect(accumulatedDuration.inMilliseconds, lessThan(160));
    });
  });

  group('AudioRecordingService State Transitions', () {
    late FakeAudioRecorder fakeRecorder;
    late Directory tempDir;
    late AudioRecordingService service;

    setUp(() async {
      fakeRecorder = FakeAudioRecorder();
      tempDir = await Directory.systemTemp.createTemp('eshitaman_test_');
      service = AudioRecordingService(
        recorder: fakeRecorder,
        pathResolver: (sessionId) async {
          final sDir = Directory('${tempDir.path}/$sessionId');
          await sDir.create(recursive: true);
          return '${sDir.path}/audio.m4a';
        },
      );
    });

    tearDown(() async {
      await service.dispose();
      if (await tempDir.exists()) {
        await tempDir.delete(recursive: true);
      }
    });

    test('transitions: idle -> recording -> paused -> recording -> stopped',
        () async {
      expect(service.state, AudioRecorderState.idle);

      // Start
      final session = await service.start();
      expect(service.state, AudioRecorderState.recording);
      expect(service.isRecording, isTrue);
      expect(fakeRecorder.startCallCount, 1);
      expect(session.status, RecordingSessionStatus.recording);

      // Pause
      await service.pause();
      expect(service.state, AudioRecorderState.paused);
      expect(service.isPaused, isTrue);
      expect(fakeRecorder.pauseCallCount, 1);

      // Resume
      await service.resume();
      expect(service.state, AudioRecorderState.recording);
      expect(service.isRecording, isTrue);
      expect(fakeRecorder.resumeCallCount, 1);

      // Stop
      final finalSession = await service.stop();
      expect(service.state, AudioRecorderState.stopped);
      expect(service.isStopped, isTrue);
      expect(fakeRecorder.stopCallCount, 1);
      expect(finalSession.status, RecordingSessionStatus.recorded);
      expect(finalSession.codec, 'aac_lc');
      expect(finalSession.mimeType, 'audio/m4a');
    });

    test('transitions: recording -> cancelled cleans up directory', () async {
      await service.start();
      expect(service.state, AudioRecorderState.recording);

      final filePath = service.currentFilePath;
      expect(filePath, isNotNull);
      final fileDir = Directory(filePath!).parent;
      expect(await fileDir.exists(), isTrue);

      await service.cancel();
      expect(service.state, AudioRecorderState.cancelled);
      expect(fakeRecorder.cancelCallCount, 1);
      expect(service.currentFilePath, isNull);
      expect(await fileDir.exists(), isFalse);
    });

    test('transitions: paused -> cancelled cleans up directory', () async {
      await service.start();
      await service.pause();
      expect(service.state, AudioRecorderState.paused);

      final filePath = service.currentFilePath;
      final fileDir = Directory(filePath!).parent;
      expect(await fileDir.exists(), isTrue);

      await service.cancel();
      expect(service.state, AudioRecorderState.cancelled);
      expect(await fileDir.exists(), isFalse);
    });
  });

  group('AudioRecordingService Defensive Error & Edge Cases', () {
    late FakeAudioRecorder fakeRecorder;
    late Directory tempDir;
    late AudioRecordingService service;

    setUp(() async {
      fakeRecorder = FakeAudioRecorder();
      tempDir = await Directory.systemTemp.createTemp('eshitaman_edge_');
      service = AudioRecordingService(
        recorder: fakeRecorder,
        pathResolver: (sessionId) async {
          final sDir = Directory('${tempDir.path}/$sessionId');
          await sDir.create(recursive: true);
          return '${sDir.path}/audio.m4a';
        },
      );
    });

    tearDown(() async {
      await service.dispose();
      if (await tempDir.exists()) {
        await tempDir.delete(recursive: true);
      }
    });

    test('pause before start is safe no-op', () async {
      expect(service.state, AudioRecorderState.idle);
      await service.pause();
      expect(service.state, AudioRecorderState.idle);
      expect(fakeRecorder.pauseCallCount, 0);
    });

    test('resume before pause is safe no-op', () async {
      expect(service.state, AudioRecorderState.idle);
      await service.resume();
      expect(service.state, AudioRecorderState.idle);
      expect(fakeRecorder.resumeCallCount, 0);
    });

    test('stop when idle throws AudioRecordingStateException', () async {
      expect(service.state, AudioRecorderState.idle);
      expect(
        () async => await service.stop(),
        throwsA(isA<AudioRecordingStateException>()),
      );
    });

    test('stop twice is idempotent and returns completed session safely',
        () async {
      await service.start();
      final session1 = await service.stop();
      expect(service.state, AudioRecorderState.stopped);

      final session2 = await service.stop();
      expect(session2.id, session1.id);
      expect(session2.status, RecordingSessionStatus.recorded);
      expect(fakeRecorder.stopCallCount, 1);
    });

    test('cancel after completed stop is safe no-op', () async {
      await service.start();
      final session = await service.stop();
      expect(service.state, AudioRecorderState.stopped);

      await service.cancel();
      // State remains stopped, not deleted
      expect(service.state, AudioRecorderState.stopped);
      expect(service.lastSession?.id, session.id);
    });

    test('start while already recording throws AudioRecordingStateException',
        () async {
      await service.start();
      expect(
        () async => await service.start(),
        throwsA(isA<AudioRecordingStateException>()),
      );
    });

    test('permission denied prevents recording and throws exception',
        () async {
      fakeRecorder.permission = false;
      expect(
        () async => await service.start(),
        throwsA(isA<AudioRecordingPermissionException>()),
      );
      expect(service.state, AudioRecorderState.idle);
      expect(service.isRecording, isFalse);
      expect(fakeRecorder.startCallCount, 0);
    });

    test('metadata.json is saved alongside audio file on stop', () async {
      await service.start();
      final session = await service.stop();
      final audioFile = File(session.filePath);
      final metaFile = File('${audioFile.parent.path}/metadata.json');

      expect(await metaFile.exists(), isTrue);
      final content = await metaFile.readAsString();
      expect(content.contains(session.id), isTrue);
      expect(content.contains('aac_lc'), isTrue);
      expect(content.contains('audio/m4a'), isTrue);
    });
  });
}
