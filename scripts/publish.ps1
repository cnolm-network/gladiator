<#
.SYNOPSIS
  Publish a curated snapshot of the Gladiator working tree to a public Git remote.

.DESCRIPTION
  Only whitelisted product files are exported. Agent workflow files
  (AGENTS.md, HANDOVER.md, TASKS.md, plans, ADRs, handover-skill, .workbuddy)
  are never copied, and an explicit check fails the run if any AGENTS.md
  appears anywhere in the export tree.

  The GitHub token is read at run time from the GIT_REPO environment variable
  (process scope first, then Windows User scope). It is:
    - never written to disk,
    - never embedded in a remote URL or stored in .git/config,
    - passed only through GIT_CONFIG_* environment variables to the git child process,
    - cleared in a finally block,
    - never printed (git output is scrubbed as a precaution).

  Credential helpers are disabled for the push, so nothing is cached.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\publish.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\publish.ps1 -NoPush
#>
param(
    [string] $SourceDir = 'D:\cbot-gladiator',
    [string] $ExportDir = 'D:\cbot-gladiator-public',
    [string] $Remote = 'https://github.com/cnolm-network/gladiator.git',
    [string] $Message = 'release: sync Gladiator public snapshot',
    [switch] $NoPush
)

$ErrorActionPreference = 'Stop'

# Product files only. Directories are copied recursively.
$include = @(
    'README.md',
    '.gitignore',
    '.editorconfig',
    'build.gradle',
    'settings.gradle',
    'gradle.properties',
    'gradlew',
    'gradlew.bat',
    'gradle/wrapper/gradle-wrapper.jar',
    'gradle/wrapper/gradle-wrapper.properties',
    'gradle/gradle-daemon-jvm.properties',
    'src/main/java',
    'src/main/resources',
    'src/test/java',
    'docs/api-contract.md',
    'docs/protocol.md',
    'docs/observation.md',
    'docs/next-steps.md',
    'docs/identity-and-action-space.md',
    'docs/production-deployment.md',
    'docs/training-env-api.md',
    'docs/training-plan-ui.md',
    'docs/scripting.md',
    'docs/models.md',
    'docs/spectating.md',
    'docs/operations.md',
    'docs/manual-testing.md',
    'docs/caller-sample.md',
    'docs/training.md',
    'bot-runtime/package.json',
    'bot-runtime/package-lock.json',
    'bot-runtime/config.example.json',
    'bot-runtime/README.md',
    'bot-runtime/src',
    'bot-runtime/scripts',
    'bot-runtime/test',
    'training/requirements.txt',
    'training/README.md',
    'training/gladiator_env',
    'training/gladiator_train',
    'training/scripts',
    'training/tests',
    'dashboard/README.md',
    'dashboard/requirements.txt',
    'dashboard/run.ps1',
    'dashboard/app.py',
    'dashboard/artifacts.py',
    'dashboard/config.py',
    'dashboard/errors.py',
    'dashboard/gmod.py',
    'dashboard/match_runner.py',
    'dashboard/models.py',
    'dashboard/model_locks.py',
    'dashboard/ports.py',
    'dashboard/presets.py',
    'dashboard/processes.py',
    'dashboard/rcon_client.py',
    'dashboard/reward_presets.py',
    'dashboard/runtime_manager.py',
    'dashboard/scripts.py',
    'dashboard/server_manager.py',
    'dashboard/terminal.py',
    'dashboard/training_runner.py',
    'dashboard/validation.py',
    'dashboard/viz.py',
    'dashboard/static',
    'dashboard/tests',
    'caller-sample/build.gradle',
    'caller-sample/src',
    'launcher/GladiatorLauncher.exe',
    'launcher/GladiatorLauncher.cs',
    'launcher/build.ps1',
    'launcher/README.md',
    'models/.gitkeep',
    'models/default.gmod',
    'models/release.gmod',
    'models/flash.gmod',
    'scripts/pack_gmod.py',
    'scripts/publish.ps1'
)

# Hard exclusions. Nothing under these paths may reach the public repo.
$forbidden = @(
    'AGENTS.md',
    'HANDOVER.md',
    'TASKS.md',
    'PROJECT_PLAN.md',
    'ARCHITECTURE.md',
    'handover-skill',
    '.workbuddy',
    '.idea',
    'server',
    'docs/ENVIRONMENT.md',
    'docs/feasibility.md',
    'docs/decisions.md',
    'docs/integration-ultrapvp.md',
    'dashboard/var',
    'training/artifacts',
    'training/.venv',
    'models/verify-ok.gmod',
    'models/verify-bad.gmod'
)

function Copy-Tree {
    param([string] $SourcePath, [string] $TargetPath)
    if (-not (Test-Path -LiteralPath $TargetPath)) {
        New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null
    }
    foreach ($child in (Get-ChildItem -LiteralPath $SourcePath -Force)) {
        $target = Join-Path $TargetPath $child.Name
        if ($child -is [System.IO.DirectoryInfo]) {
            Copy-Tree -SourcePath $child.FullName -TargetPath $target
        } else {
            Copy-Item -LiteralPath $child.FullName -Destination $target -Force
        }
    }
}

$token = $null
try {
    # --- 1. Prepare export tree -------------------------------------------
    # Never delete from the export tree: the host's safe-delete guard blocks
    # recursive removals. Files are only copied/overwritten. Section 2 fails
    # the run if an unexpected file shows up, so nothing stale can be pushed.
    if (Test-Path -LiteralPath $ExportDir) {
        if (-not (Test-Path -LiteralPath (Join-Path $ExportDir '.git'))) {
            throw "Export directory exists but is not a publish repo: $ExportDir"
        }
    } else {
        New-Item -ItemType Directory -Path $ExportDir -Force | Out-Null
    }

    # Copy file-by-file so an existing target directory never gets a nested
    # copy of itself (Copy-Item -Recurse would create e.g. java/java/...).
    foreach ($item in $include) {
        $src = Join-Path $SourceDir $item
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Whitelisted path is missing: $item"
        }
        $dst = Join-Path $ExportDir $item
        $parent = Split-Path -Parent $dst
        if (-not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        if ((Get-Item -LiteralPath $src) -is [System.IO.DirectoryInfo]) {
            Copy-Tree -SourcePath $src -TargetPath $dst
        } else {
            Copy-Item -LiteralPath $src -Destination $dst -Force
        }
    }

    # --- 2. Verify no agent/internal files and no token leak --------------
    foreach ($item in $forbidden) {
        if (Test-Path -LiteralPath (Join-Path $ExportDir $item)) {
            throw "Forbidden path present in export: $item"
        }
    }
    $agentFiles = @(Get-ChildItem -LiteralPath $ExportDir -Recurse -Force -File -Filter 'AGENTS.md' -ErrorAction SilentlyContinue)
    if ($agentFiles.Count -gt 0) {
        throw "AGENTS.md found in export tree ($($agentFiles.Count) file(s))"
    }

    # Since nothing is deleted, verify the export tree is exactly the whitelist.
    $expected = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($item in $include) {
        $full = Join-Path $SourceDir $item
        if (Test-Path -LiteralPath $full -PathType Container) {
            Get-ChildItem -LiteralPath $full -Recurse -File -Force | ForEach-Object {
                $rel = $_.FullName.Substring($SourceDir.Length).TrimStart('\', '/').Replace('\', '/')
                [void] $expected.Add($rel)
            }
        } else {
            [void] $expected.Add($item.Replace('\', '/'))
        }
    }
    $extras = @(Get-ChildItem -LiteralPath $ExportDir -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\\.git\\' } |
        Where-Object {
            $rel = $_.FullName.Substring($ExportDir.Length).TrimStart('\', '/').Replace('\', '/')
            -not $expected.Contains($rel)
        })
    if ($extras.Count -gt 0) {
        throw "Export tree has $($extras.Count) file(s) outside the whitelist; clean them manually before publishing: $($extras[0].FullName)"
    }

    # --- 3. Commit the snapshot -------------------------------------------
    if (-not (Test-Path -LiteralPath (Join-Path $ExportDir '.git'))) {
        & git -C $ExportDir init -q -b main
        if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
    }
    & git -C $ExportDir add -A
    if ($LASTEXITCODE -ne 0) { throw 'git add failed' }

    $changes = @(& git -C $ExportDir status --porcelain)
    if ($changes.Count -gt 0) {
        $name = (& git -C $SourceDir config user.name)
        $email = (& git -C $SourceDir config user.email)
        if ([string]::IsNullOrWhiteSpace($name)) { $name = 'CnolmLab' }
        if ([string]::IsNullOrWhiteSpace($email)) { $email = 'noreply@github.com' }
        & git -C $ExportDir -c "user.name=$name" -c "user.email=$email" commit -q -m $Message
        if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
        Write-Output "Committed $($changes.Count) change(s) to the public snapshot."
    } else {
        Write-Output 'No file changes in the public snapshot.'
    }

    if ($NoPush) {
        Write-Output 'NoPush specified: snapshot prepared locally, nothing sent.'
        return
    }

    # --- 4. Resolve token at run time only --------------------------------
    $token = $env:GIT_REPO
    if ([string]::IsNullOrWhiteSpace($token)) {
        $token = [System.Environment]::GetEnvironmentVariable('GIT_REPO', 'User')
    }
    if ([string]::IsNullOrWhiteSpace($token)) {
        throw 'GIT_REPO token is not available in process or User environment.'
    }

    # Also verify no exported text file accidentally contains the token.
    $leaks = @(Get-ChildItem -LiteralPath $ExportDir -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\\.git\\' } |
        Where-Object {
            $text = [System.IO.File]::ReadAllText($_.FullName)
            $text.Contains($token)
        })
    if ($leaks.Count -gt 0) {
        throw "Token string found in $($leaks.Count) exported file(s); aborting."
    }

    # --- 5. Push with credentials only in process environment ------------
    $basic = [Convert]::ToBase64String(
        [System.Text.Encoding]::ASCII.GetBytes('x-access-token:' + $token))
    # An empty GIT_CONFIG_VALUE_* is rejected by git, so credential.helper is
    # cleared with a command-line -c instead.
    $env:GIT_CONFIG_COUNT = '1'
    $env:GIT_CONFIG_KEY_0 = 'http.extraHeader'
    $env:GIT_CONFIG_VALUE_0 = "AUTHORIZATION: basic $basic"
    $env:GIT_TERMINAL_PROMPT = '0'

    # Windows certificate store (the sandbox proxy terminates TLS) and Git
    # protocol v0 (the proxy returns 502 on the protocol-v2 upload-pack POST).
    # These are transport settings only; no credential is stored.
    $gitTransport = @(
        '-c', 'credential.helper=',
        '-c', 'http.sslBackend=schannel',
        '-c', 'http.schannelCheckRevoke=false',
        '-c', 'protocol.version=0'
    )
    # git writes normal progress to stderr; with ErrorActionPreference=Stop
    # PowerShell would turn those lines into terminating NativeCommandError.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # GET-only check: avoid a needless POST when the remote already matches.
        # The sandbox proxy intermittently answers POSTs with 502.
        $refs = (& git -C $ExportDir @gitTransport ls-remote --heads $Remote 2>&1) -join "`n"
        $localSha = (& git -C $ExportDir rev-parse HEAD).Trim()
        $remoteSha = ''
        if ($refs -match '([0-9a-f]{40})\s+refs/heads/main') { $remoteSha = $Matches[1] }
        if ($remoteSha -and $remoteSha -eq $localSha) {
            Write-Output "Remote main already matches $localSha; nothing to push."
            return
        }

        $code = 0
        $out = @()
        # One bounded retry for a transient proxy 502; never use --force.
        for ($attempt = 1; $attempt -le 2; $attempt++) {
            $out = & git -C $ExportDir @gitTransport push $Remote main 2>&1
            $code = $LASTEXITCODE
            if ($code -eq 0) { break }
            if ($attempt -eq 1 -and ($out -join "`n") -match '502') {
                Write-Output 'Transient proxy 502; retrying once.'
                Start-Sleep -Seconds 2
                continue
            }
            break
        }
    } finally {
        $ErrorActionPreference = $previous
    }

    $out | ForEach-Object {
        $line = $_.ToString()
        if ($token) { $line = $line.Replace($token, '***') }
        if ($basic) { $line = $line.Replace($basic, '***') }
        Write-Output $line
    }
    if ($code -ne 0) {
        throw "git push failed with exit code $code (remote may contain unrelated history; no force push was attempted)"
    }
    Write-Output 'Push complete.'
}
finally {
    # --- 6. Scrub every trace from this process ---------------------------
    foreach ($key in @('GIT_CONFIG_COUNT', 'GIT_CONFIG_KEY_0', 'GIT_CONFIG_VALUE_0',
                       'GIT_TERMINAL_PROMPT')) {
        [System.Environment]::SetEnvironmentVariable($key, $null, 'Process')
    }
    $basic = $null
    $token = $null
    [System.GC]::Collect()
}
