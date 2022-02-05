" move cursor everyedit; add whitespace if needed when editing
set virtualedit=all

" Don't try to be vi compatible
set nocompatible

" Helps force plugins to load correctly when it is turned back on below
filetype off

" TODO: Load plugins here (pathogen or vundle)

" Turn on syntax highlighting
syntax on

" For plugins to load correctly
filetype plugin indent on
"let g:tex_flavor="latex"

" TODO: Pick a leader key
" let mapleader = ","

" Security
set modelines=0

" Show line numbers
set number

" Show file stats
set ruler

" Blink cursor on error instead of beeping (grr)
set visualbell

" Encoding
set encoding=utf-8

" Show carriage returns
set list

" Whitespace
set wrap
set linebreak
"set textwidth=0
"set wrapmargin=0
"set formatoptions=tcqrn1
set tabstop=4
set shiftwidth=4
set softtabstop=4
set smarttab
set expandtab
"set noshiftround
set autoindent
set smartindent
set showmatch

" Cursor motion
set scrolloff=16
set backspace=indent,eol,start
set matchpairs+=<:> " use % to jump between pairs
"runtime! macros/matchit.vim

" Move up/down editor lines
nnoremap j gj
nnoremap k gk

" Allow hidden buffers i.e. quit vim without being
" prompted about unsaved hidden buffers that have
" been modified.
set hidden

" Rendering
set ttyfast

" Status bar
set laststatus=2

" Last line
set showmode
set showcmd

set ignorecase

" Formatting
"map <leader>q gqip

" Visualize tabs and newlines
"set listchars=tab:▸\ ,eol:¬
" Uncomment this to enable by default:
" set list " To enable by default
" Or use your leader key + l to toggle on/off
map <leader>l :set list!<CR> " Toggle tabs and EOL

set undodir=~/.vim/undodir
set undofile

set incsearch
set hlsearch

set cursorline
set cursorcolumn

set foldmethod=indent
set foldnestmax=3
set nofoldenable
set foldlevel=20

" open NERDTree automatically when vim starts
autocmd vimenter * NERDTree  | wincmd p

" close vim if only window left is NERDTree
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif

" remap NERDTree command
nnoremap <leader>n :NERDTreeFocus<CR>
nnoremap <C-n> :NERDTree<CR>
nnoremap <C-t> :NERDTreeToggle<CR>
nnoremap <C-f> :NERDTreeFind<CR>


" turn off search highlighting with control-l
nnoremap <silent> <C-l> :<C-u>nohlsearch<CR>

" delete buffer but keep window open (switching to alternative buffer)
"nnoremap <C-c> :bp\|bd #<CR>

" delete buffer (and move to previous buffer) without closing tab
"noremap <leader>p :bp \| bd # <return>
"noremap <leader>n :bn \| bd # <return>


" remove any trailing whitespace on save
autocmd BufWrite * :%s/\s\+$//e

"
" Color scheme theme
set background=dark
set termguicolors
colorscheme gruvbox

" gruvbox theme https://github.com/morhetz/gruvbox/wiki/Installation
autocmd vimenter * ++nested colorscheme gruvbox
let g:gruvbox_contrast_dark = 'hard' " options are: soft, medium, hard
let g:gruvbox_contrast_light = 'soft' " options are: soft, medium, hard
let g:gruvbox_improved_strings = 1
let g:gruvbox_improved_warnings = 1
let g:gruvbox_invert_indent_guides = 0
let g:gruvbox_invert_selection = 1
let g:gruvbox_invert_signs = 1
let g:gruvbox_invert_tabline = 1
