\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nthu_thesis}[2025/07/01 NTHU Library DSLIN]

\LoadClass[12pt,openany,oneside]{book}

\usepackage{CJKutf8}

\usepackage[encoding=UTF8]{zhnumber}

\usepackage{pdfpages}

% === Enforce Symmetric Margins ===
\RequirePackage{geometry}

\usepackage{emptypage}  %prevents page numbers and headings from appearing on empty pages.
  
% \usepackage{showframe} % for viewing margins 
\usepackage{times}


% Allow 10pt/11pt/12pt font size options
\DeclareOption{10pt}{\def\@ptsize{0}}
\DeclareOption{11pt}{\def\@ptsize{1}}
\DeclareOption{12pt}{\def\@ptsize{2}}

\ExecuteOptions{12pt}  % Default to 12pt if not specified
\ProcessOptions\relax

% Load appropriate font size settings
\input{size1\@ptsize.clo}


\usepackage{setspace}
\usepackage[utf8]{inputenc}

\setstretch{1.5}  % gives MS Word-like 1.5x spacing across fonts 
% \onehalfspacing % ~1.25x spacing only 

\AtBeginEnvironment{thebibliography}{\singlespacing} % Bibliography
\AtBeginEnvironment{figure}{\singlespacing} % Figure captions
\AtBeginEnvironment{table}{\singlespacing} % Table captions


\usepackage{ragged2e} % for line alignment justify options

\usepackage{indentfirst}
\usepackage[skip=10pt plus1pt, indent=2em]{parskip}


\usepackage{amsmath, amsthm, amssymb}   % improve math presentation

\usepackage{booktabs}                   % 改善表格設置
%\usepackage{tabularx} % extra features for tabular environment

\usepackage{multirow}
\setlength{\arrayrulewidth}{0.3mm} %sets the thickness of the borders of the table. In the example is 0.5mm but you can use other units—see the article Lengths in LaTeX for a complete list.
\setlength{\tabcolsep}{18pt} %The space between the text and the left/right border of its containing cell is set to 18pt with this command. Again, you may use other units if needed.
\renewcommand{\arraystretch}{1.5} %The height of each row is set to 1.5 relative to its default height.

\usepackage{titlesec}  %for title spacing
\titleformat{\chapter}[hang] 
{\normalfont\LARGE\bfseries\centering}{\chaptertitlename\ \thechapter}{1em}{} 
%\titleformat{\section}{\large\bfseries\MakeUppercase}{\thesection}{1em}{}

\renewcommand{\chaptername}{Chapter}

%\titlespacing*{⟨command⟩}{⟨left⟩}{⟨before-sep⟩}{⟨after-sep⟩}[⟨right-sep⟩]
%\titlespacing*{\chapter}{0pt}{-20pt}{15pt}
\titlespacing*{\chapter}
   {0pt}% left
   {-1em}% before
   {1em}% after

%\titlespacing\section{0pt}{14pt plus 4pt minus 2pt}{0pt plus 2pt minus 2pt}
%\titlespacing\subsection{0pt}{13pt plus 4pt minus 2pt}{0pt plus 2pt minus 2pt}



\usepackage[allow-number-unit-breaks]{siunitx}


\usepackage{romannum} %\romannum{...} turns an Arabic number into a lowercase Roman one
% scalefnt：本機 TeX Live 2026 basic 無此 package，且主文未用 \scalefont{}，先停用。
% \usepackage{scalefnt}

\usepackage{nomencl} % for symbols
\makenomenclature  %needed to couple with above line nomencl
% glossaries：本機環境需 sudo 才能安裝（package non-relocatable），且 Abbreviations
% 表已用 tabularx 手寫，沒有用 \gls。若日後改用 glossaries 機制，再 uncomment 並
% 安裝即可（tlmgr install glossaries）。
% \usepackage{glossaries}
% \makeglossaries

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{graphicx}     
\graphicspath{{./figures/}}%% define absolute path to images %%

%\usepackage{background} % for full page image setting
\usepackage{caption}  % caption width follows the figure/table (container) width, not a fixed 0.75 fraction
% Caption settings
\usepackage[labelformat=simple]{subcaption}
\captionsetup[figure]{name=Fig., font=normalsize, labelfont=bf} % 10pt captions, bold label % 

\captionsetup[table]{font=normalsize, labelfont=bf} % Apply to tables if needed

\usepackage{wrapfig}  
\usepackage[labelformat=simple]{subcaption} %subfigure captions need this

\usepackage{gensymb}
\usepackage[absolute,overlay]{textpos}

%\usepackage{enumitem} % Ensure proper handling of enumerate environments
%\setlength{\itemindent}{-2em}

% Chapter format in TOC
% Preamble and settings
\usepackage{tocloft}  % Added for TOC customization
%\usepackage{titletoc}
% Configure TOC title format with tocloft
% set font size of chapter titles in TOC
%\renewcommand{\cftchapfont}{\large\bfseries} % Larger, bold chapter titles in TOC
\renewcommand{\cftchapfont}{\normalfont\bfseries}  % \chapter font in ToC

\setlength{\cftbeforetoctitleskip}{1\baselineskip} % vertical Space before TOC title "Contents"
\setlength{\cftaftertoctitleskip}{1\baselineskip} % vertical Space after TOC title

%\renewcommand\cftchapafterpnum{\vskip-5pt}
%\renewcommand\cftsecafterpnum{\vskip-10pt}



% TOC title formatting
%\renewcommand{\cfttoctitlefont}{\hspace*{\fill}\large\bfseries}
\renewcommand{\cftaftertoctitle}{\hspace*{\fill}}


\setcounter{tocdepth}{2}     % set toc depth
\setcounter{secnumdepth}{2}   % set sec numbering depth
% Insert table of contents
%\setcounter{tocdepth}{-1} -- only parts
%\setcounter{tocdepth}{0} -- only parts and chapters
%\setcounter{tocdepth}{1} -- part,chapters,sections
%\setcounter{tocdepth}{2} -- part,chapters,sections, subsections


 % Options
\DeclareOption{master}{\gdef\@degreeZH{碩}}
\DeclareOption{doctor}{\gdef\@degreeZH{博}}
%\DeclareOption{nowatermark}{\gdef\nowatermark{1}}
\ProcessOptions\relax   %must be inserted after all the option-handling commands were typed.

% \appendix{NUM}{TITLE}
%
\usepackage[titletoc,toc,title]{appendix}

\usepackage{blindtext} % to generate dummy text for testing

% biblatex backend：biber（homebrew，2026-07-17 起本機已裝 v2.21，對應 biblatex v3.21）。
% 先前因 TeX Live 2026 basic 未內建 biber 而暫用 bibtex backend，副作用是 biblatex 每次
% 都發 "Using fall-back bibtex backend" + "There were undefined references" 假警報。
\usepackage[style=ieee]{biblatex}

\addbibresource{back/references.bib}
\usepackage[english]{babel}


\usepackage{color}		% For colored text
\usepackage{xcolor}
\usepackage[colorlinks=true, urlcolor=blue, linkcolor=red]{hyperref}
%\usepackage[final]{hyperref} % adds hyper links inside the generated pdf file
\hypersetup{colorlinks=true,linkcolor=black,citecolor=blue,filecolor=magenta,urlcolor=blue}