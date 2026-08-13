Unicode True
RequestExecutionLevel user

!include "MUI2.nsh"
!include "LogicLib.nsh"

!ifndef APP_VERSION
  !error "APP_VERSION must be supplied with /DAPP_VERSION=X.Y.Z"
!endif
!ifndef OUTPUT_FILE
  !error "OUTPUT_FILE must be supplied with /DOUTPUT_FILE=path"
!endif

Name "DPTK"
OutFile "${OUTPUT_FILE}"
InstallDir "$LOCALAPPDATA\Programs\DPTK"
InstallDirRegKey HKCU "Software\DPTK\Installer" "InstallLocation"
BrandingText "DPTK ${APP_VERSION}"
SetCompressor /SOLID lzma

VIProductVersion "${APP_VERSION}.0"
VIAddVersionKey "ProductName" "DPTK"
VIAddVersionKey "ProductVersion" "${APP_VERSION}"
VIAddVersionKey "FileDescription" "DPTK Installer"

!define MUI_ABORTWARNING
!define MUI_ICON "..\assets\icons\app_icon.ico"
!define MUI_UNICON "..\assets\icons\app_icon.ico"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "DPTK (required)" SecMain
  SectionIn RO
  SetShellVarContext current
  SetOutPath "$INSTDIR"
  File /r "..\dist\PySide6_Data_Plot\*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\DPTK"
  CreateShortcut "$SMPROGRAMS\DPTK\DPTK.lnk" "$INSTDIR\PySide6_Data_Plot.exe"
  CreateShortcut "$SMPROGRAMS\DPTK\Uninstall DPTK.lnk" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\DPTK\Installer" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DPTK" "DisplayName" "DPTK"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DPTK" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DPTK" "Publisher" "ylylyl98"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DPTK" "DisplayIcon" "$INSTDIR\PySide6_Data_Plot.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DPTK" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DPTK" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DPTK" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DPTK" "NoRepair" 1
SectionEnd

Section /o "Desktop shortcut" SecDesktop
  SetShellVarContext current
  CreateShortcut "$DESKTOP\DPTK.lnk" "$INSTDIR\PySide6_Data_Plot.exe"
  WriteRegDWORD HKCU "Software\DPTK\Installer" "DesktopShortcut" 1
SectionEnd

Section "Uninstall"
  SetShellVarContext current
  ReadRegDWORD $0 HKCU "Software\DPTK\Installer" "DesktopShortcut"
  ${If} $0 == 1
    Delete "$DESKTOP\DPTK.lnk"
  ${EndIf}
  RMDir /r "$SMPROGRAMS\DPTK"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DPTK"
  DeleteRegKey HKCU "Software\DPTK\Installer"
  RMDir /r "$INSTDIR"
SectionEnd
