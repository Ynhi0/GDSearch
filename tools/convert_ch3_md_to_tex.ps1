$mdPath = 'final_deliverables\reports\CHAPTER_3_METHODODOLOGY.md'
$outPath = 'final_deliverables\reports\CHAPTER_3_METHODODOLOGY.tex'

$bytes = [System.IO.File]::ReadAllBytes($mdPath)
$text = [System.Text.Encoding]::UTF8.GetString($bytes)
$text = $text -replace "`r`n", "`n"

function Normalize-Plain {
  param([string]$s)
  $formD = $s.Normalize([System.Text.NormalizationForm]::FormD)
  $sb = New-Object System.Text.StringBuilder
  foreach ($ch in $formD.ToCharArray()) {
    if ([Globalization.CharUnicodeInfo]::GetUnicodeCategory($ch) -ne 'NonSpacingMark') {
      [void]$sb.Append($ch)
    }
  }
  return $sb.ToString()
}

$plain = Normalize-Plain $text
$headingKey = '## Tai lieu tham khao (IEEE)'
$idx = $plain.IndexOf($headingKey, [System.StringComparison]::OrdinalIgnoreCase)
if ($idx -ge 0) {
  $lineEnd = $text.IndexOf("`n", $idx)
  if ($lineEnd -lt 0) { $lineEnd = $text.Length }
  $body = $text.Substring(0, $idx).Trim()
  $refs = $text.Substring($lineEnd).Trim()
} else {
  $body = $text.Trim()
  $refs = ''
}

function Convert-Inline {
  param([string]$s)
  $s = [regex]::Replace($s, '\*\*(.+?)\*\*', '\textbf{$1}')
  $s = [regex]::Replace($s, '(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', '\emph{$1}')
  $s = [regex]::Replace($s, '`([^`]+)`', '\texttt{$1}')
  return $s
}

function Expand-Range {
  param([int]$a, [int]$b)
  $list = New-Object System.Collections.Generic.List[string]
  if ($a -le $b) {
    for ($i = $a; $i -le $b; $i++) { $list.Add("ref$i") }
  } else {
    for ($i = $a; $i -ge $b; $i--) { $list.Add("ref$i") }
  }
  return ($list -join ',')
}

function Convert-Citations {
  param([string]$s)
  $s = [regex]::Replace(
    $s,
    '\[(\d+)\]\s*(?:\p{Pd}|-)\s*\[(\d+)\]',
    { param($m) $a = [int]$m.Groups[1].Value; $b = [int]$m.Groups[2].Value; "\cite{" + (Expand-Range $a $b) + "}" }
  )
  $s = [regex]::Replace($s, '\[(\d+)\]', { param($m) "\cite{ref" + $m.Groups[1].Value + "}" })
  return $s
}

function Convert-Equations {
  param([string]$s)
  return [regex]::Replace(
    $s,
    '\$\$(.+?)\$\$',
    { param($m) "\begin{equation}`n" + ($m.Groups[1].Value.Trim()) + "`n\end{equation}" },
    [System.Text.RegularExpressions.RegexOptions]::Singleline
  )
}

function Convert-Headings {
  param([string]$s)
  $s = [regex]::Replace($s, '^###\s+(.+)$', '\subsubsection{$1}', [System.Text.RegularExpressions.RegexOptions]::Multiline)
  $s = [regex]::Replace($s, '^##\s+(.+)$', '\subsection{$1}', [System.Text.RegularExpressions.RegexOptions]::Multiline)
  $s = [regex]::Replace($s, '^#\s+(.+)$', '\section{$1}', [System.Text.RegularExpressions.RegexOptions]::Multiline)
  return $s
}

function Escape-Alg {
  param([string]$s)
  $s = $s -replace '\\', '\textbackslash{}'
  $s = $s -replace '([%#&$])', '\$1'
  $s = $s -replace '_', '\_'
  $s = $s -replace '\{', '\{'
  $s = $s -replace '\}', '\}'
  $s = $s -replace [char]0x2208, '$\in$'
  $s = $s -replace [char]0x222A, '$\cup$'
  return $s
}

function Escape-Bib {
  param([string]$s)
  $urls = New-Object System.Collections.Generic.List[string]
  $s = [regex]::Replace($s, '\\url\{[^}]*\}', { param($m) $urls.Add($m.Value); "{URL$($urls.Count-1)}" })
  $s = $s -replace '([%#&$])', '\$1'
  $s = $s -replace '_', '\_'
  for ($i = 0; $i -lt $urls.Count; $i++) {
    $s = $s.Replace("{URL$i}", $urls[$i])
  }
  return $s
}

$segments = $body -split '```'
$sb = New-Object System.Text.StringBuilder
for ($i = 0; $i -lt $segments.Length; $i++) {
  if (($i % 2) -eq 0) {
    $seg = $segments[$i]
    $seg = Convert-Equations $seg
    $seg = Convert-Headings $seg
    $seg = Convert-Inline $seg
    $seg = Convert-Citations $seg
    [void]$sb.Append($seg)
  } else {
    $block = $segments[$i] -replace "`r`n", "`n"
    $lines = $block -split "`n"
    if ($lines.Length -gt 0 -and ($lines[0].Trim().ToLower() -in @('text', 'plaintext', ''))) {
      if ($lines.Length -gt 1) { $lines = $lines[1..($lines.Length - 1)] } else { $lines = @() }
    }
    if ($lines.Length -eq 0) { continue }
    $captionText = 'Mo ta thuat toan'
    if ($lines.Length -gt 0) {
      $firstNorm = (Normalize-Plain $lines[0]).ToLowerInvariant()
      if ($firstNorm -like 'thuat toan*' -and $lines[0] -match ':\s*(.+)$') {
        $captionText = $matches[1].Trim()
        if ($lines.Length -gt 1) { $lines = $lines[1..($lines.Length - 1)] } else { $lines = @() }
      }
    }

    $requires = @()
    $ensures = @()
    $steps = @()
    foreach ($ln in $lines) {
      $t = $ln.Trim()
      if ($t -eq '') { continue }
      $norm = (Normalize-Plain $t).ToLowerInvariant()
      if ($norm -like 'dau vao*' -and $t -match ':\s*(.+)$') { $requires += $matches[1].Trim(); continue }
      if ($norm -like 'dau ra*' -and $t -match ':\s*(.+)$') { $ensures += $matches[1].Trim(); continue }
      if ($t -match '^(Input)\s*:\s*(.+)$') { $requires += $matches[2].Trim(); continue }
      if ($t -match '^(Output)\s*:\s*(.+)$') { $ensures += $matches[2].Trim(); continue }
      $steps += $ln
    }

    $alg = New-Object System.Text.StringBuilder
    [void]$alg.Append("`n\begin{algorithm}[H]`n\caption{$captionText}`n\begin{algorithmic}[1]`n")
    foreach ($r in $requires) { [void]$alg.Append("\Require $(Escape-Alg $r)`n") }
    foreach ($e in $ensures) { [void]$alg.Append("\Ensure $(Escape-Alg $e)`n") }
    foreach ($st in $steps) {
      $line = $st -replace '^\s*\d+\s*:\s*', ''
      $indent = ($line -replace '^(\s*).*$','$1').Length
      $content = Escape-Alg ($line.TrimStart())
      if ($content -eq '') { continue }
      if ($indent -ge 2) {
        [void]$alg.Append("\State \hspace*{1em} $content`n")
      } else {
        [void]$alg.Append("\State $content`n")
      }
    }
    [void]$alg.Append("\end{algorithmic}`n\end{algorithm}`n")
    [void]$sb.Append($alg.ToString())
  }
}

$bodyLatex = $sb.ToString()

$bib = ''
if ($refs.Trim() -ne '') {
  $refMatches = [regex]::Matches($refs, '\[(\d+)\]\s+(.*?)(?=\n\s*\[\d+\]\s+|\z)', [System.Text.RegularExpressions.RegexOptions]::Singleline)
  $bibSB = New-Object System.Text.StringBuilder
  [void]$bibSB.Append("\begin{thebibliography}{99}`n")
  foreach ($m in $refMatches) {
    $num = $m.Groups[1].Value
    $entry = $m.Groups[2].Value.Trim()
    $entry = $entry -replace '\s*`n\s*', ' '
    $entry = Convert-Inline $entry
    $entry = [regex]::Replace($entry, 'Available:\s*(https?://\S+)', { param($mm) $url = $mm.Groups[1].Value.TrimEnd('.', ',', ';'); "Available: \url{$url}" })
    $entry = Escape-Bib $entry
    [void]$bibSB.Append("\bibitem{ref$num} $entry`n")
  }
  [void]$bibSB.Append("\end{thebibliography}`n")
  $bib = $bibSB.ToString()
}

$preamble = @'
\documentclass[a4paper,12pt]{article}
\usepackage[a4paper,left=3cm,right=2cm,top=2cm,bottom=2cm]{geometry}
\usepackage{iftex}
\ifPDFTeX
\PackageError{CHAPTER_3_METHODODOLOGY}{This report must be compiled with XeLaTeX}{Run xelatex CHAPTER_3_METHODODOLOGY.tex}
\fi
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{unicode-math}
\setmathfont{Cambria Math}
\usepackage{setspace}
\setstretch{1.35}
\usepackage{indentfirst}
\setlength{\parindent}{1.25em}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{float}
\usepackage{enumitem}
\usepackage[unicode,hidelinks]{hyperref}
\usepackage{xurl}
\Urlmuskip=0mu plus 1mu
\renewcommand{\refname}{Tài liệu tham khảo (IEEE)}
\newcommand{\ind}{\mathbf{1}}
% Vietnamese algorithm keywords
\floatname{algorithm}{Thuật toán}
\algrenewcommand\algorithmicrequire{\textbf{Đầu vào:}}
\algrenewcommand\algorithmicensure{\textbf{Đầu ra:}}
\algrenewcommand\algorithmicreturn{\textbf{Trả về:}}
\algrenewcommand\algorithmicfor{\textbf{Đối với mỗi}}
\algrenewcommand\algorithmicforall{\textbf{Với mỗi}}
\algrenewcommand\algorithmicif{\textbf{Nếu}}
\algrenewcommand\algorithmicelse{\textbf{Ngược lại}}
\algrenewcommand\algorithmicthen{\textbf{thì}}
\algrenewcommand\algorithmicdo{\textbf{thực hiện}}
\algrenewcommand\algorithmicwhile{\textbf{Trong khi}}
\algrenewcommand\algorithmicrepeat{\textbf{Lặp}}
\algrenewcommand\algorithmicuntil{\textbf{cho đến khi}}
\begin{document}

'@

$latex = $preamble + $bodyLatex + "`n" + $bib + "`n\end{document}`n"
[System.IO.File]::WriteAllText($outPath, $latex, [System.Text.Encoding]::UTF8)
