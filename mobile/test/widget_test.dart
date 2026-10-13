import 'package:flutter_test/flutter_test.dart';
import 'package:eshitaman/main.dart';

void main() {
  testWidgets('App smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const EshitamanApp());
    expect(find.byType(EshitamanApp), findsOneWidget);
  });
}
