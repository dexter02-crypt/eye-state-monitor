"""Local model-file guards using fabricated bytes. NEVER actual MediaPipe inference."""
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch, Mock
import local_support as s

class ModelFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.root_patch = patch.object(s, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def seed(self, data=b'fabricated model bytes, not a runnable model'):
        name, url = s.MODELS['face']
        path = self.root/'models'/name
        s.write_new(path, data)
        s.write_json(path.with_suffix('.receipt.json'), {'url':url, 'bytes':len(data),
                       'sha256':hashlib.sha256(data).hexdigest()})
        return path

    def test_missing_model_is_explicit_failure(self):
        with self.assertRaises(ValueError): s.model_path('face')

    def test_matching_local_receipt_checks_bytes_only(self):
        path = self.seed()
        self.assertEqual(s.model_path('face'), path)

    def test_tampered_bytes_are_not_accepted(self):
        path = self.seed()
        path.write_bytes(b'changed model file')
        with self.assertRaises(ValueError): s.model_path('face')

    def test_non_object_receipt_rejected(self):
        path = self.seed()
        path.with_suffix('.receipt.json').write_text('[]')
        with self.assertRaises(ValueError): s.model_path('face')

    def test_wrong_source_receipt_rejected(self):
        path = self.seed()
        receipt = s.read_json(path.with_suffix('.receipt.json'))
        receipt['url'] = 'https://example.invalid/not-model'
        import json
        path.with_suffix('.receipt.json').write_text(json.dumps(receipt))
        with self.assertRaises(ValueError): s.model_path('face')

    def test_model_symlink_rejected(self):
        path = self.seed()
        original = path.with_suffix('.backup')
        path.rename(original)
        path.symlink_to(original)
        with self.assertRaises(ValueError): s.model_path('face')

    def test_bad_download_never_saves_a_model(self):
        opener = Mock()
        opener.open.return_value = io.BytesIO(b'not a zip archive')
        with patch.object(s.urllib.request, 'build_opener', return_value=opener):
            with self.assertRaises(ValueError): s.download_model('face')
        self.assertFalse((self.root/'models'/s.MODELS['face'][0]).exists())

    def test_mock_download_records_received_bytes_and_reuses(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('fixture.tflite', b'fabricated payload, not runnable')
        data = stream.getvalue()
        opener = Mock()
        opener.open.return_value = io.BytesIO(data)
        with patch.object(s.urllib.request, 'build_opener', return_value=opener):
            path = s.download_model('face')
        receipt = s.read_json(path.with_suffix('.receipt.json'))
        self.assertEqual(receipt['sha256'], hashlib.sha256(data).hexdigest())
        self.assertIn('not an upstream signature', receipt['verification'])
        with patch.object(s.urllib.request, 'build_opener', side_effect=AssertionError('No network expected')):
            self.assertEqual(s.download_model('face'), path)
