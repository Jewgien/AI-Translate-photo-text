import json
import sys
import time
from decimal import Decimal
from pathlib import Path
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from rdflib import Graph, Literal, URIRef, RDF
from rdflib.namespace import OWL
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

BASE = Path(__file__).resolve().parent
OWL_FILE = BASE / "ocr_translate.owl"      
YOLO_FILE = BASE / "best.pt"               
THR_DETECTOR = 0.50                        
THR_OCR = 0.80                             
MODELS = {"ru": "Helsinki-NLP/opus-mt-ru-en", "en": "Helsinki-NLP/opus-mt-en-ru"}  

def load_image(path):
    p = Path(path)
    if not p.is_file():
        raise ValueError("Файл не найден.")
    if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError("Поддерживаются только файлы JPEG и PNG.")
    if p.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Файл больше 10 МБ.")
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Файл повреждён или не является изображением.")
    return img


def text_kind(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "num"
    return "ok" if all(c.isascii() or "\u0400" <= c <= "\u04ff" for c in letters) else "other"


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def assemble(frags):
    words = sorted((f for f in frags if f["text"].strip()),
                   key=lambda f: ((f["box"][1] + f["box"][3]) / 2, f["box"][0]))
    lines = []                                           
    for w in words:
        yc, h = (w["box"][1] + w["box"][3]) / 2, w["box"][3] - w["box"][1]
        if lines and abs(yc - lines[-1]["yc"]) < 0.5 * max(h, lines[-1]["h"]):
            lines[-1]["words"].append(w)
        else:
            lines.append({"words": [w], "yc": yc, "h": h})
    groups = []                                          
    for ln in lines:
        ln["words"].sort(key=lambda w: w["box"][0])
        ln["box"] = union([w["box"] for w in ln["words"]])
        if groups:
            prev = groups[-1][-1]["box"]
            gap = ln["box"][1] - prev[3]
            overlap = min(prev[2], ln["box"][2]) - max(prev[0], ln["box"][0])
            if gap < 1.2 * ln["h"] and overlap > 0:
                groups[-1].append(ln)
                continue
        groups.append([ln])
    blocks = []
    for grp in groups:
        ws = [w for ln in grp for w in ln["words"]]
        blocks.append({"text": " ".join(w["text"].strip() for w in ws),
                       "box": union([w["box"] for w in ws]),
                       "low": any(w["low"] for w in ws), "translation": ""})
    return blocks

class Knowledge:
    NAMES = ["Изображение", "Текстовая_область", "Распознанный_фрагмент",
             "Содержит_область", "Распознается_как", "распознанныйТекст", "уверенностьOCR"]

    def __init__(self, path):
        self.base = Graph()
        data = Path(path).read_bytes()
        if not data.strip():
            raise ValueError("Файл онтологии пустой: пересохраните её в Protégé (File → Save as → RDF/XML Syntax)")
        self.base.parse(data=data, format="xml")            
        cls = next((s for s in self.base.subjects(RDF.type, OWL.Class)
                    if str(s).endswith("Изображение")), None)
        if cls is None:
            raise ValueError("В онтологии нет класса «Изображение»")
        self.ns = str(cls)[:-len("Изображение")]            
        for name in self.NAMES:
            if (self.t(name), None, None) not in self.base:
                raise ValueError(f"В онтологии нет «{name}»")
        self.g = self.base

    def t(self, name):
        return URIRef(self.ns + name)

    def add(self, frags):
        self.g = Graph()
        self.g += self.base
        img = URIRef(self.ns + "img_1")
        self.g.add((img, RDF.type, self.t("Изображение")))
        for i, f in enumerate(frags):
            area, frag = URIRef(f"{self.ns}area_{i}"), URIRef(f"{self.ns}frag_{i}")
            self.g.add((area, RDF.type, self.t("Текстовая_область")))
            self.g.add((frag, RDF.type, self.t("Распознанный_фрагмент")))
            self.g.add((img, self.t("Содержит_область"), area))
            self.g.add((area, self.t("Распознается_как"), frag))
            self.g.add((frag, self.t("распознанныйТекст"), Literal(f["text"])))
            self.g.add((frag, self.t("уверенностьOCR"), Literal(Decimal(f"{f['conf']:.3f}"))))

    def low_confidence(self, thr):
        query = f"""
            SELECT ?frag ?text ?conf WHERE {{
                ?frag a <{self.t('Распознанный_фрагмент')}> ;
                      <{self.t('распознанныйТекст')}> ?text ;
                      <{self.t('уверенностьOCR')}> ?conf .
                FILTER (?conf < {thr})
            }}"""
        rows = [(int(str(r.frag).rsplit("_", 1)[1]), str(r.text), float(r.conf.toPython()))
                for r in self.g.query(query)]
        return sorted(rows)

    def mean_confidence(self):
        query = f"SELECT (AVG(?c) AS ?m) WHERE {{ ?f <{self.t('уверенностьOCR')}> ?c . }}"
        value = next(iter(self.g.query(query)))[0]
        return float(value.toPython()) if value is not None else 0.0


def num(x):
    return f"{x:.2f}".replace(".", ",")


class Bridge:
    def __init__(self):
        self.kb = Knowledge(OWL_FILE)
        self.yolo = self.reader = None                 
        self.mt = {}

    def load_models(self, say):
        if self.yolo is None:
            say("Загрузка моделей (первый раз долго)…")
            try:
                yolo = YOLO(str(YOLO_FILE))
            except Exception as e:
                raise RuntimeError(f"Не удалось открыть веса детектора best.pt: файл повреждён или скачан не полностью. Скачайте его заново.\n{e}")
            try:
                reader = easyocr.Reader(["ru", "en"], verbose=False)
            except Exception as e:
                raise RuntimeError(f"Не удалось загрузить модели EasyOCR: удалите папку .EasyOCR\\model в вашей папке пользователя и запустите заново.\n{e}")
            self.yolo, self.reader = yolo, reader

    def process(self, path, src, mode, say=lambda text: None):
        t0 = time.time()
        res = {"path": path, "src": src, "mode": mode, "error": "", "img": None, "blocks": [],
               "log": [], "tips": [], "unreliable": False, "message": "", "ms": 0}
        try:
            res["img"] = load_image(path)
            self.load_models(say)
            self.run(res, say)
        except Exception as e:                         
            res["error"] = str(e)
        res["ms"] = int((time.time() - t0) * 1000)
        return res

    def run(self, res, say):
        img, mode, log, tips = res["img"], res["mode"], res["log"], res["tips"]

        say("Поиск текстовых областей…")
        boxes = self.detect(img, THR_DETECTOR)
        if not boxes and mode == "precise":            
            boxes = self.detect(img, 0.25, scale=2)
            log.append(("R7", "областей нет, режим «Точный» → повторный поиск с порогом 0,25 на увеличенном снимке"))
        if not boxes:                                  
            log.append(("R01", "текстовые области не найдены → результат пуст"))
            tips.append("Текст не обнаружен: подойдите ближе, улучшите освещение, наведите фокус.")
            res["message"] = "Текст не обнаружен"
            return

        say("Распознавание текста…")
        frags = [self.read(img, b) for b in boxes]
        self.kb.add(frags)
        low = self.kb.low_confidence(THR_OCR)          
        if low:                                        
            names = ", ".join(f"«{t[:20]}»" for _, t, _ in low)
            log.append(("R02", f"уверенность фрагментов {names} ниже {num(THR_OCR)} "
                               f"(минимум {num(min(c for _, _, c in low))}) → выделены для проверки"))
            if mode == "precise":                      
                log.append(("R03", "режим «Точный», повтор не выполнялся → увеличение, резкость и повторный OCR"))
                for i, _, _ in low:
                    better = self.read(img, frags[i]["box"], sharpen=True)
                    if better["conf"] > frags[i]["conf"]:
                        frags[i] = better
                self.kb.add(frags)
                low = self.kb.low_confidence(THR_OCR)
                if low:                                
                    log.append(("R04", "после повтора уверенность всё ещё ниже порога → предложена ручная правка"))
                    tips.append("Повтор не помог: исправьте текст в поле «Оригинал» и нажмите «Перевести заново».")
            else:                                      
                log.append(("F08", "режим «Интерактивный»: повторные проходы запрещены → добавлена рекомендация"))
                tips.append("Для сомнительных фрагментов включите режим «Точный».")
        for i, _, _ in low:
            frags[i]["low"] = True

        res["blocks"] = assemble(frags)
        if not res["blocks"]:
            log.append(("R01", "области найдены, но текст не распознан → результат пуст"))
            tips.append("Текст не распознан: переснимите изображение ближе и при хорошем освещении.")
            res["message"] = "Текст не распознан"
            return

        share, mean = len(low) / len(frags), self.kb.mean_confidence()
        if share > 0.30 or mean < 0.50:                
            res["unreliable"] = True
            log.append(("R15", f"ненадёжных фрагментов {share * 100:.0f} %, средняя уверенность {num(mean)} → результат недостоверен"))
            tips.append("Результат недостоверен: переснимите изображение (свет, расстояние, ракурс, фокус).")

        say("Перевод…")
        try:
            done = self.translate([b["text"] for b in res["blocks"]], res["src"], log)
            for b, t in zip(res["blocks"], done):
                b["translation"] = t
        except Exception as e:                         
            log.append(("R06", f"переводчик недоступен ({e}) → сохранён результат распознавания"))
            tips.append("Переводчик недоступен. Позже нажмите «Перевести заново».")
        res["message"] = ("Результат недостоверен" if res["unreliable"] else
                          "Есть сомнительные фрагменты — проверьте текст" if low else "Готово")

    def detect(self, img, conf, scale=1):
        big = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        r = self.yolo.predict(big, conf=conf, imgsz=640, verbose=False)[0]
        boxes = [tuple(int(v / scale) for v in b) for b in r.boxes.xyxy.tolist()]
        return sorted(boxes, key=lambda b: (b[1], b[0]))

    def read(self, img, box, sharpen=False):
        x1, y1, x2, y2 = box
        crop = img[max(y1 - 3, 0):y2 + 3, max(x1 - 3, 0):x2 + 3]
        if sharpen:
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            crop = cv2.addWeighted(crop, 1.5, cv2.GaussianBlur(crop, (0, 0), 3), -0.5, 0)
        h, w = crop.shape[:2]
        parts = self.reader.readtext(crop) or self.reader.recognize(crop, [[0, w, 0, h]], [])
        parts = [p for p in parts if p[1].strip()]
        if not parts:
            return {"box": box, "text": "", "conf": 0.0, "low": False}
        return {"box": box, "text": " ".join(p[1].strip() for p in parts),
                "conf": float(sum(p[2] for p in parts) / len(parts)), "low": False}

    def translate(self, texts, src, log=None):
        out, todo = [], []
        for i, text in enumerate(texts, 1):
            kind = text_kind(text) if text.strip() else "empty"
            if kind == "ok":
                out.append(None)
                todo.append(i - 1)
                continue
            out.append(text if kind == "num" else "")
            if log is not None and kind in ("num", "other"):
                code = "R12" if kind == "num" else "R05"
                log.append((code, f"блок {i} не переводится ({'цифры и знаки' if kind == 'num' else 'язык не поддерживается'})"))
        if todo:
            for i, t in zip(todo, self.marian([texts[i] for i in todo], src)):
                out[i] = t
        return out

    def marian(self, texts, src):
        if src not in self.mt:
            from transformers import MarianMTModel, MarianTokenizer
            self.mt[src] = (MarianTokenizer.from_pretrained(MODELS[src]), MarianMTModel.from_pretrained(MODELS[src]))
        tok, model = self.mt[src]
        batch = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        out = model.generate(**batch, num_beams=4, max_new_tokens=256)
        return tok.batch_decode(out, skip_special_tokens=True)

    def save(self, res, folder):
        stem = Path(res["path"]).stem
        txt, js = Path(folder) / f"{stem}_result.txt", Path(folder) / f"{stem}_result.json"
        txt.write_text("\n\n".join(f"[{i}] {b['text']}\n    -> {b['translation']}"
                                   for i, b in enumerate(res["blocks"], 1)), encoding="utf-8")
        data = {"image": Path(res["path"]).name, "mode": res["mode"], "time_ms": res["ms"],
                "blocks": [{"text": b["text"], "translation": b["translation"], "low": b["low"]} for b in res["blocks"]],
                "recommendations": res["tips"],
                "trace": [{"rule": c, "explanation": e} for c, e in res["log"]]}
        js.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return txt, js

def to_pixmap(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


def draw_boxes(res):
    img = res["img"].copy()
    t = max(2, img.shape[1] // 400)
    for i, b in enumerate(res["blocks"], 1):
        if b["box"] is None:
            continue
        color = (0, 140, 255) if b["low"] else (0, 160, 0)
        x1, y1, x2, y2 = b["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
        cv2.putText(img, str(i), (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, t * 0.4, color, t)
    return img


def button(text, handler):
    b = QPushButton(text)
    b.clicked.connect(lambda checked=False: handler())
    return b


def layout(kind, *items):
    box = kind()
    for it in items:
        box.addWidget(it) if isinstance(it, QWidget) else box.addLayout(it)
    return box


def title(text):
    label = QLabel(text)
    label.setStyleSheet("font-size: 20px; font-weight: bold;")
    return label


class Worker(QThread):
    step = Signal(str)
    done = Signal(object)

    def __init__(self, bridge, path, src, mode):
        super().__init__()
        self.args = (bridge, path, src, mode)

    def run(self):
        bridge, path, src, mode = self.args
        self.done.emit(bridge.process(path, src, mode, self.step.emit))


class Window(QMainWindow):
    def __init__(self, bridge):
        super().__init__()
        self.bridge, self.res, self.path = bridge, None, ""
        self.setWindowTitle("OCR-Translate")
        self.resize(1100, 680)
        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)
        for page in (self.page_input(), self.page_result(), self.page_explain()):
            self.pages.addWidget(page)

    def page_input(self):
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("Файл не выбран")
        self.preview = QLabel("Предпросмотр изображения")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(380)
        self.lang = QComboBox()
        self.lang.addItems(["ru → en", "en → ru"])
        self.mode = QComboBox()
        self.mode.addItems(["Интерактивный", "Точный"])
        self.status = QLabel("")
        page = QWidget()
        page.setLayout(layout(
            QVBoxLayout, title("Распознавание и перевод текста"),
            layout(QHBoxLayout, self.file_edit, button("Открыть…", self.choose_file)),
            self.preview,
            layout(QHBoxLayout, QLabel("Языки:"), self.lang, QLabel("Режим:"), self.mode),
            layout(QHBoxLayout, button("Распознать и перевести", self.on_run), button("Очистить", self.on_clear)),
            self.status))
        return page

    def page_result(self):
        self.picture = QLabel()
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setMinimumSize(520, 400)
        self.original = QPlainTextEdit()
        self.translation = QPlainTextEdit()
        self.translation.setReadOnly(True)
        self.tips = QLabel()
        self.tips.setWordWrap(True)
        self.result_status = QLabel()
        right = layout(QVBoxLayout, QLabel("Оригинал (один блок в строке, можно править)"), self.original,
                       QLabel("Перевод"), self.translation)
        page = QWidget()
        page.setLayout(layout(
            QVBoxLayout, title("Результат распознавания и перевода"),
            layout(QHBoxLayout, self.picture, right),
            QLabel("Рекомендации"), self.tips,
            layout(QHBoxLayout, button("Перевести заново", self.on_retranslate),
                   button("Сохранить TXT / JSON", self.on_save), button("Пояснение", self.on_explain),
                   button("Новое изображение", self.on_clear), self.result_status)))
        return page

    def page_explain(self):
        self.rules_text = QPlainTextEdit()
        self.rules_text.setReadOnly(True)
        page = QWidget()
        page.setLayout(layout(QVBoxLayout, title("Пояснение результата"), self.rules_text,
                              button("Назад", lambda: self.pages.setCurrentIndex(1))))
        return page

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Изображения (*.jpg *.jpeg *.png)")
        if path:
            self.path = path
            self.file_edit.setText(path)
            self.preview.setPixmap(QPixmap(path).scaled(
                700, 380, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def on_run(self):
        if not self.path:
            QMessageBox.information(self, "OCR-Translate", "Сначала откройте изображение.")
            return
        self.start("precise" if self.mode.currentIndex() == 1 else "interactive")

    def start(self, mode):
        src = "ru" if self.lang.currentIndex() == 0 else "en"
        self.status.setText("Запуск…")
        self.worker = Worker(self.bridge, self.path, src, mode)
        self.worker.step.connect(self.status.setText)
        self.worker.done.connect(self.show_result)
        self.worker.start()

    def show_result(self, res):
        self.status.setText("")
        if res["error"]:
            QMessageBox.warning(self, "Ошибка", res["error"])
            return
        self.res = res
        self.picture.setPixmap(to_pixmap(draw_boxes(res)).scaled(
            520, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.original.setPlainText("\n".join(b["text"] for b in res["blocks"]))
        self.translation.setPlainText("\n".join(b["translation"] for b in res["blocks"]))
        self.tips.setText("\n".join("• " + t for t in res["tips"]) or "Замечаний нет.")
        self.result_status.setText(f"{res['message']} · {res['ms']} мс")
        self.pages.setCurrentIndex(1)
        if res["unreliable"] and res["mode"] != "precise":      
            answer = QMessageBox.question(self, "Результат недостоверен", "Повторить обработку в точном режиме?")
            if answer == QMessageBox.StandardButton.Yes:
                self.start("precise")

    def on_retranslate(self):
        lines = [t.strip() for t in self.original.toPlainText().split("\n")]
        try:
            new = self.bridge.translate(lines, self.res["src"], self.res["log"])
        except Exception as e:
            QMessageBox.warning(self, "Переводчик недоступен", f"{e}\nРезультат распознавания сохранён.")
            return
        blocks = self.res["blocks"]
        for i, (text, tr) in enumerate(zip(lines, new)):
            if i >= len(blocks):
                blocks.append({"text": text, "translation": tr, "low": False, "box": None})
            else:
                if text != blocks[i]["text"]:
                    blocks[i]["low"] = False
                blocks[i]["text"], blocks[i]["translation"] = text, tr
        del blocks[len(lines):]
        self.translation.setPlainText("\n".join(new))

    def on_save(self):
        folder = QFileDialog.getExistingDirectory(self, "Куда сохранить результат")
        if folder:
            txt, js = self.bridge.save(self.res, folder)
            QMessageBox.information(self, "Сохранено", f"{txt}\n{js}")

    def on_explain(self):
        lines = [f"{i}. {code}: {text}" for i, (code, text) in enumerate(self.res["log"], 1)]
        self.rules_text.setPlainText("\n".join(lines) or "Правила не срабатывали.")
        self.pages.setCurrentIndex(2)

    def on_clear(self):
        self.path = ""
        self.file_edit.clear()
        self.preview.setText("Предпросмотр изображения")
        self.pages.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = Window(Bridge())
    except Exception as e:               
        QMessageBox.critical(None, "OCR-Translate", f"Не удалось запустить: {e}")
        sys.exit(1)
    window.show()
    sys.exit(app.exec())
