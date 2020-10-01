mkdir -p ~/.streamlit/
echo “\
[general]\n\
email = \arbozaleon@gmail.com\”\n\
“ > ~/.streamlit/credentials.toml
echo “\
[server]\n\
Headless = true\n\
enableCORS = false\n\
port = $PORT\n\
“ > ~/.streamlit/config.toml
