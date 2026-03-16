import re

from bs4 import BeautifulSoup


START_MARK = r"<!-- Начало разворачивающегося блока -->"
END_MARK = r"<!-- Конец разворачивающегося блока -->"
BLOCKS_PATTERN = re.compile(f"{START_MARK}(.*?){END_MARK}", re.DOTALL)

SORRY_MARK = "Приносим извинения за доставленные неудобства."


def extract_blocks(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 0. Поиск названия страницы
    title = parse_title(html)

    # 1. Обработка части ДО первого разворачивающегося блока
    first_start = re.search(START_MARK, html)
    if first_start:
        before_html = html[:first_start.start()]
    else:
        before_html = html

    upper_part = "\n".join([
        part for parts in parse_text_containers(before_html, with_title=False)
        if SORRY_MARK not in (part:=" ".join(parts))
    ])

    # 2. Поиск всех разворачивающихся блоков
    matches = BLOCKS_PATTERN.findall(html)

    lower_part = []
    for block_html in matches:
        parsed_block_title, parsed_block = parse_expandable_block(block_html)
        if parsed_block:
            lower_part.append((parsed_block_title, parsed_block))

    return title, upper_part, lower_part


def parse_title(html):
    soup = BeautifulSoup(html, "lxml")
    
    title_block = soup.find("title")
    if title_block:
        title = title_block.get_text(strip=False)
        return postprocess_text(title)
    
    return ""


def parse_text_containers(html, with_title=False):
    soup = BeautifulSoup(html, "lxml")
    
    blocks = []

    containers = soup.find_all(class_="text-container")
    for container in containers:
        lines = []

        if with_title:
            title = container.find(class_="title-base")
            if title:
                lines.append(title.get_text(strip=False))

        paragraphs = container.find_all("p")
        for p in paragraphs:
            text = p.get_text(strip=False)
            if text and (cleaned:=postprocess_text(text)):
                lines.append(cleaned)

        if lines:
            blocks.append(lines)

    return blocks


def parse_expandable_block(html):
    soup = BeautifulSoup(html, "lxml")
    
    title, lines = "", []

    # title из button → title-base
    button = soup.find("button")
    if button:
        title_block = button.find(class_="title-base")
        if title_block:
            title = postprocess_text(
                title_block.get_text(strip=False)
            )

    # текст из text-container
    container = soup.find(class_="text-container")
    if container:
        paragraphs = container.find_all("p")
        for p in paragraphs:
            text = p.get_text(strip=False)
            if text and (cleaned:=postprocess_text(text)):
                lines.append(cleaned)

    return title, "\n".join(lines)


def postprocess_text(text):
    return re.sub(r"\s+", " ", text).strip()




if __name__ == "__main__":
    title, description_part, info_part = extract_blocks("data/Меры поддержки жителей блокадного Ленинграда.html")
