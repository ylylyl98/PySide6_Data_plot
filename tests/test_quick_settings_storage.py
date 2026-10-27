"""Native registry paths cannot become on-disk workspace or cache paths."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import os
import subprocess
import sys
import textwrap

from PySide6.QtCore import QSettings
from ui_qt.quick.settings_storage import settings_storage_dir,child_settings_file,appearance_preference,analysis_session_path


class SettingsStorageTests(unittest.TestCase):
    def test_native_registry_filename_is_never_used_as_a_filesystem_path(self):
        class NativeSettings:
            def format(self):return QSettings.Format.NativeFormat
            def fileName(self):raise AssertionError("Native registry filename must not be read as a path")
        with TemporaryDirectory() as temporary:
            target=Path(temporary).resolve()/"app-data"
            with patch("ui_qt.quick.settings_storage.QStandardPaths.writableLocation",return_value=str(target)) as writable:
                self.assertEqual(settings_storage_dir(NativeSettings()),target.resolve())
            with patch.dict(os.environ,LOCALAPPDATA=temporary):
                for kind,title in (("peak","Peak Analysis"),("drr","DRR Analysis")):
                    expected=Path(temporary).resolve()/"DPTK"/title/"workspace.npz"
                    self.assertEqual(analysis_session_path(NativeSettings(),kind),expected.resolve())
                    self.assertEqual(analysis_session_path(None,kind),expected.resolve())
            self.assertFalse(target.exists(),"Resolving a storage path must not create directories")
            writable.assert_called_once()

    def test_explicit_ini_keeps_all_services_beside_the_injected_file(self):
        class IniSettings:
            def __init__(self,path):self.path=path
            def format(self):return QSettings.Format.IniFormat
            def fileName(self):return self.path
        with TemporaryDirectory() as temporary:
            path=Path(temporary).resolve()/"isolated"/"quick.ini"
            with patch("ui_qt.quick.settings_storage.QStandardPaths.writableLocation",side_effect=AssertionError("INI must remain isolated")):
                self.assertEqual(settings_storage_dir(IniSettings(str(path))),path.parent.resolve())
                for kind,subdir in (("peak","peak-analysis"),("drr","drr-analysis")):
                    self.assertEqual(analysis_session_path(IniSettings(str(path)),kind),path.parent/subdir/"workspace.npz")
            self.assertFalse(path.parent.exists())

    def test_child_ini_copies_legacy_theme_without_writing_native_settings(self):
        class NativeSettings:
            def format(self):return QSettings.Format.NativeFormat
            def fileName(self):return r"\HKEY_CURRENT_USER\Software\DPTK\PySide6_Data_Plot"
            def value(self,key,default=None):return {"ui/appearance":"dark"}.get(key,default)
            def setValue(self,*args):raise AssertionError("Parent native settings must not be written")
        with TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()/"application-data"
            with patch("ui_qt.quick.settings_storage.QStandardPaths.writableLocation",return_value=str(root)):
                destination=child_settings_file(NativeSettings(),"mcd-organizer")
            self.assertEqual(destination,root/"mcd-organizer"/"quick.ini")
            child=QSettings(str(destination),QSettings.Format.IniFormat)
            self.assertEqual(child.value("quick/appearance"),"dark")
            self.assertNotIn("HKEY",str(destination));self.assertEqual(list(root.rglob("*.ini")),[destination])

    def test_missing_native_app_data_fails_before_any_current_directory_write(self):
        class NativeSettings:
            def format(self):return QSettings.Format.NativeFormat
        with patch("ui_qt.quick.settings_storage.QStandardPaths.writableLocation",return_value=""):
            with self.assertRaises(RuntimeError):settings_storage_dir(NativeSettings())

    def test_current_theme_precedes_legacy_theme(self):
        class Settings:
            def value(self,key,default=None):return {"ui/appearance":"dark","quick/appearance":"light"}.get(key,default)
        self.assertEqual(appearance_preference(Settings()),"light")

    def test_real_services_use_native_app_data_and_do_not_read_registry_as_a_filename(self):
        code=textwrap.dedent('''
            from pathlib import Path
            import sys
            from unittest.mock import patch
            from PySide6.QtCore import QSettings
            from PySide6.QtGui import QGuiApplication
            from ui_qt.quick.peak_analysis_launcher import QuickPeakAnalysisLauncher
            from ui_qt.quick.drr_analysis_launcher import QuickDrrAnalysisLauncher
            from ui_qt.quick.drr_analysis_bridge import DrrAnalysisBridge
            from ui_qt.quick.presentation_bridge import PresentationBridge
            class NativeSettings:
                def format(self): return QSettings.Format.NativeFormat
                def fileName(self): raise AssertionError('Registry filename cannot be a disk location')
                def value(self,key,default=None,*args,**kwargs): return default
                def setValue(self,*args): raise AssertionError('Audit cannot write native settings')
            app=QGuiApplication([])
            base=Path(sys.argv[1]).resolve()
            with patch('ui_qt.quick.settings_storage.QStandardPaths.writableLocation',return_value=str(base)):
                peak=QuickPeakAnalysisLauncher(settings=NativeSettings())
                drr=QuickDrrAnalysisLauncher(settings=NativeSettings())
                analysis=DrrAnalysisBridge(settings=NativeSettings())
                slides=PresentationBridge(settings=NativeSettings())
                assert peak.session_path==base/'DPTK'/'Peak Analysis'/'workspace.npz'
                assert drr.directory==base/'DPTK'/'DRR Analysis'
                assert analysis.storage_base==base/'DPTK'/'DRR Analysis'
                assert slides._cache_root==base/'slides-catalog'
                peak.close();drr.close();analysis.close();slides.close()
            assert 'PySide6.QtWidgets' not in sys.modules
        ''')
        with TemporaryDirectory() as temporary:
            env=dict(os.environ,QT_QPA_PLATFORM="offscreen",QT_QUICK_BACKEND="software",LOCALAPPDATA=temporary,MPLCONFIGDIR=temporary,XDG_CONFIG_HOME=temporary)
            result=subprocess.run([sys.executable,"-c",code,temporary],env=env,cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_default_cli_checks_do_not_read_or_write_native_settings_or_production_workspaces(self):
        code=textwrap.dedent('''
            from pathlib import Path
            import importlib,sys
            import PySide6.QtCore as qt
            original=qt.QSettings
            class IsolatedSettings(original):
                def __init__(self,*args,**kwargs):
                    assert len(args)>=2 and args[1]==original.IniFormat, 'Check accessed native user settings'
                    super().__init__(*args,**kwargs)
            qt.QSettings=IsolatedSettings
            module=importlib.import_module(sys.argv[1])
            raise SystemExit(module.main(['--check']))
        ''')
        for module in ("run_peak_analysis_quick","run_drr_analysis_quick"):
            with self.subTest(module=module),TemporaryDirectory() as temporary:
                root=Path(temporary).resolve()
                for title in ("Peak Analysis","DRR Analysis"):
                    production=root/"DPTK"/title/"workspace.npz"
                    production.parent.mkdir(parents=True)
                    production.write_bytes(b"Protected preexisting production workspace")
                before={str(path.relative_to(root)):path.read_bytes() for path in root.rglob("*") if path.is_file()}
                env=dict(os.environ,QT_QPA_PLATFORM="offscreen",QT_QUICK_BACKEND="software",LOCALAPPDATA=temporary,
                         MPLCONFIGDIR=str(root/"mpl"),XDG_CONFIG_HOME=str(root/"config"),PYTHONIOENCODING="utf-8")
                result=subprocess.run([sys.executable,"-c",code,module],env=env,
                    cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertNotIn("TypeError:",result.stderr,"Successful exit must not hide QML teardown errors")
                after={str(path.relative_to(root)):path.read_bytes() for path in (root/"DPTK").rglob("*") if path.is_file()}
                self.assertEqual(after,before,"Check mode changed production workspace/settings files")


if __name__=="__main__":unittest.main()
