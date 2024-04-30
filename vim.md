:set number           # Show and Hide Line Numbers

# https://stackoverflow.com/questions/18948491/running-python-code-in-vim
`:w !python`

`autocmd FileType python map <buffer> <F9> :w<CR>:exec '!python' shellescape(@%, 1)<CR>`
`autocmd FileType python imap <buffer> <F9> <esc>:w<CR>:exec '!python' shellescape(@%, 1)<CR>`

Explanation:

- `autocmd:` command that Vim will execute automatically on `{event}` (here: if you open a python file)
- `[i]map`: creates a keyboard shortcut to `<F9>` in insert/normal mode
- `<buffer>`: If multiple buffers/files are open: just use the active one
- `<esc>`: leaving insert mode
- `:w<CR>`: saves your file
- `!`: runs the following command in your shell (try `:!ls`)
- `%`: is replaced by the filename of your active buffer. But since it can contain things like whitespace and other "bad" stuff it is better practise not to write `:python %`, but use:
- `shellescape`: escape the special characters. The `1` means with a backslash



