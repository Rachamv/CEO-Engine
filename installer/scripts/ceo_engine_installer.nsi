; CEO Engine — Windows Installer Script
; =======================================
; Build with NSIS 3.x:
;   makensis installer/scripts/ceo_engine_installer.nsi
;
; Produces: CEO_Engine_Setup_v3.6.0.exe
; Installs to: C:\Program Files\CEO Engine\
; Creates: Desktop shortcut, Start Menu entry, Add/Remove Programs entry
; Requires: dist\CEOEngine\ folder from PyInstaller build

;----------------------------------
; General
;----------------------------------
!define APP_NAME        "CEO Engine"
!define APP_VERSION     "3.6.0"
!define APP_PUBLISHER   "Rachamv"
!define APP_URL         "http://localhost:5000"
!define APP_EXE         "CEOEngine.exe"
!define INSTALL_DIR     "$PROGRAMFILES64\CEO Engine"
!define REG_KEY         "Software\Microsoft\Windows\CurrentVersion\Uninstall\CEOEngine"
!define MUI_ICON        "..\assets\icon.ico"
!define MUI_UNICON      "..\assets\icon.ico"

Name "${APP_NAME} ${APP_VERSION}"
OutFile "..\..\CEO_Engine_Setup_v3.6.0.exe"
InstallDir "${INSTALL_DIR}"
InstallDirRegKey HKLM "${REG_KEY}" "InstallLocation"
RequestExecutionLevel admin
SetCompressor /SOLID lzma
Unicode True

;----------------------------------
; Modern UI
;----------------------------------
!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "WinMessages.nsh"

!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE      "Welcome to CEO Engine ${APP_VERSION}"
!define MUI_WELCOMEPAGE_TEXT       "CEO Engine is a professional algorithmic trading system that connects to MetaTrader 5.$\r$\n$\r$\nThis wizard will install CEO Engine on your computer.$\r$\n$\r$\nClick Next to continue."
!define MUI_FINISHPAGE_RUN         "$INSTDIR\${APP_EXE}"
!define MUI_FINISHPAGE_RUN_TEXT    "Launch CEO Engine now"
!define MUI_FINISHPAGE_SHOWREADME  "$INSTDIR\QUICKSTART.md"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Open Quick Start guide"
!define MUI_FINISHPAGE_LINK        "Visit documentation"
!define MUI_FINISHPAGE_LINK_LOCATION "https://github.com/Rachamv/ceo-engine"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE       "..\..\README.md"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Languages
!insertmacro MUI_LANGUAGE "English"

;----------------------------------
; Version info embedded in .exe
;----------------------------------
VIProductVersion "3.6.0.0"
VIAddVersionKey "ProductName"      "${APP_NAME}"
VIAddVersionKey "ProductVersion"   "${APP_VERSION}"
VIAddVersionKey "CompanyName"      "${APP_PUBLISHER}"
VIAddVersionKey "FileDescription"  "CEO Engine Installer"
VIAddVersionKey "FileVersion"      "${APP_VERSION}"
VIAddVersionKey "LegalCopyright"   "© 2024 ${APP_PUBLISHER}"

;----------------------------------
; Component descriptions
;----------------------------------
InstType "Full (recommended)"
InstType "Compact"

Section "CEO Engine (required)" SecCore
    SectionIn RO    ; cannot be deselected
    SectionIn 1 2

    SetOutPath "$INSTDIR"

    ; Copy all bundled files from PyInstaller dist/CEOEngine/
    File /r "..\..\dist\CEOEngine\*.*"

    ; Write uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; Write registry keys for Add/Remove Programs
    WriteRegStr   HKLM "${REG_KEY}" "DisplayName"          "${APP_NAME}"
    WriteRegStr   HKLM "${REG_KEY}" "DisplayVersion"        "${APP_VERSION}"
    WriteRegStr   HKLM "${REG_KEY}" "Publisher"             "${APP_PUBLISHER}"
    WriteRegStr   HKLM "${REG_KEY}" "InstallLocation"       "$INSTDIR"
    WriteRegStr   HKLM "${REG_KEY}" "UninstallString"       "$INSTDIR\Uninstall.exe"
    WriteRegStr   HKLM "${REG_KEY}" "DisplayIcon"           "$INSTDIR\${APP_EXE},0"
    WriteRegStr   HKLM "${REG_KEY}" "URLInfoAbout"          "${APP_URL}"
    WriteRegDWORD HKLM "${REG_KEY}" "NoModify"              1
    WriteRegDWORD HKLM "${REG_KEY}" "NoRepair"              1
    ; Compute install size for Add/Remove Programs
    ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
    IntFmt $0 "0x%08X" $0
    WriteRegDWORD HKLM "${REG_KEY}" "EstimatedSize" "$0"

SectionEnd

Section "Desktop shortcut" SecDesktop
    SectionIn 1
    CreateShortcut "$DESKTOP\CEO Engine.lnk" \
        "$INSTDIR\${APP_EXE}" "" \
        "$INSTDIR\${APP_EXE}" 0 \
        SW_SHOWNORMAL "" "CEO Engine Trading System"
SectionEnd

Section "Start Menu shortcuts" SecStartMenu
    SectionIn 1 2
    CreateDirectory "$SMPROGRAMS\CEO Engine"
    CreateShortcut "$SMPROGRAMS\CEO Engine\CEO Engine.lnk" \
        "$INSTDIR\${APP_EXE}" "" \
        "$INSTDIR\${APP_EXE}" 0 \
        SW_SHOWNORMAL "" "Launch CEO Engine"
    CreateShortcut "$SMPROGRAMS\CEO Engine\CEO Engine (CLI).lnk" \
        "$INSTDIR\ceo-run.exe" "" \
        "$INSTDIR\${APP_EXE}" 0 \
        SW_SHOWNORMAL "" "CEO Engine command-line runner"
    CreateShortcut "$SMPROGRAMS\CEO Engine\Quick Start Guide.lnk" \
        "$INSTDIR\QUICKSTART.md"
    CreateShortcut "$SMPROGRAMS\CEO Engine\Uninstall.lnk" \
        "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Auto-start with Windows" SecAutostart
    SectionIn 1
    WriteRegStr HKCU \
        "Software\Microsoft\Windows\CurrentVersion\Run" \
        "CEOEngine" \
        '"$INSTDIR\${APP_EXE}"'
SectionEnd

;----------------------------------
; Section descriptions (shown on hover in component page)
;----------------------------------
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${SecCore}       "Core CEO Engine files, dashboard server, and MT5 connector. Required."
    !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop}    "Add a CEO Engine shortcut to your desktop."
    !insertmacro MUI_DESCRIPTION_TEXT ${SecStartMenu}  "Add CEO Engine to the Start Menu."
    !insertmacro MUI_DESCRIPTION_TEXT ${SecAutostart}  "Start CEO Engine automatically when Windows starts (recommended for prop firm traders)."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;----------------------------------
; Installer logic
;----------------------------------
Function .onInit
    ; Check if already installed — offer to uninstall first
    ReadRegStr $R0 HKLM "${REG_KEY}" "UninstallString"
    StrCmp $R0 "" done

    MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION \
        "CEO Engine is already installed.$\n$\nClick OK to remove the previous version before installing ${APP_VERSION}." \
        IDOK uninst
    Abort

    uninst:
        ClearErrors
        ExecWait '$R0 /S _?=$INSTDIR'
        IfErrors no_remove_uninstaller
        no_remove_uninstaller:
    done:
FunctionEnd

Function .onInstSuccess
    ; Brief pause so progress bar completes visually
    Sleep 500
FunctionEnd

;----------------------------------
; Uninstaller
;----------------------------------
Section "Uninstall"

    ; Remove all installed files
    RMDir /r "$INSTDIR"

    ; Remove shortcuts
    Delete "$DESKTOP\CEO Engine.lnk"
    RMDir /r "$SMPROGRAMS\CEO Engine"

    ; Remove auto-start entry
    DeleteRegValue HKCU \
        "Software\Microsoft\Windows\CurrentVersion\Run" \
        "CEOEngine"

    ; Remove registry entry
    DeleteRegKey HKLM "${REG_KEY}"

    ; Note: user data (ceo_engine_config.json, ceo_journal.db, logs)
    ; is stored in the install folder which is now deleted.
    ; We intentionally do NOT delete AppData to preserve trade journal.

SectionEnd
