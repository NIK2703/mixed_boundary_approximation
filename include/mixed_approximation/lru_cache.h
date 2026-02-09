#ifndef MIXED_APPROXIMATION_LRU_CACHE_H
#define MIXED_APPROXIMATION_LRU_CACHE_H

#include <unordered_map>
#include <list>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace mixed_approx {

/**
 * @brief Базовый шаблон кэша с политикой вытеснения
 * 
 * @tparam Key тип ключа
 * @tparam Value тип значения
 */
template<typename Key, typename Value>
class CacheBase {
public:
    virtual ~CacheBase() = default;
    
    virtual void clear() = 0;
    virtual std::size_t size() const noexcept = 0;
    virtual bool empty() const noexcept = 0;
};

/**
 * @brief LRU-кэш (Least Recently Used)
 * 
 * Комбинация std::unordered_map для O(1) доступа и 
 * std::list для отслеживания порядка использования.
 * 
 * Поддержка квантования ключей для вещественных чисел:
 * близкие значения x считаются идентичными для кэширования.
 * 
 * @tparam Key тип ключа (должен поддерживать хэширование)
 * @tparam Value тип значения
 */
template<typename Key, typename Value>
class LRUCache : public CacheBase<Key, Value> {
private:
    using ListIterator = typename std::list<std::pair<Key, Value>>::iterator;
    
    std::list<std::pair<Key, Value>> items_;  // список (key, value) в порядке использования
    std::unordered_map<Key, ListIterator> map_; // быстрый доступ по ключу
    std::size_t max_size_;
    
    // Квантование для вещественных чисел
    bool use_quantization_;
    double quantization_step_;
    double interval_length_;
    
    /**
     * @brief Квантование ключа для вещественных чисел
     */
    Key quantize_key(const Key& key) const {
        if (!use_quantization_) return key;
        // Специализация для double
        if constexpr (std::is_same_v<Key, double>) {
            if (interval_length_ <= 0) return key;
            double normalized = key / interval_length_;
            double quantized_normalized = std::round(normalized / quantization_step_) * quantization_step_;
            return quantized_normalized * interval_length_;
        }
        return key;
    }
    
    /**
     * @brief Вытеснение элемента при переполнении
     */
    void evict_if_needed() {
        while (items_.size() > max_size_) {
            items_.pop_front();
        }
    }
    
    /**
     * @brief Перемещение элемента в конец (недавно использованный)
     */
    void move_to_back(ListIterator it) {
        if (it != items_.end()) {
            items_.splice(items_.end(), items_, it);
        }
    }
    
public:
    /**
     * @brief Конструктор LRU-кэша
     * @param max_size максимальное количество элементов
     * @param enable_quantization включить квантование ключей
     * @param quantization_threshold порог квантования (относительный)
     */
    explicit LRUCache(std::size_t max_size = 1024, 
                      bool enable_quantization = true,
                      double quantization_threshold = 1e-9)
        : max_size_(max_size)
        , use_quantization_(enable_quantization)
        , quantization_step_(quantization_threshold)
        , interval_length_(1.0) {}
    
    /**
     * @brief Установка длины интервала для квантования
     * @param length длина интервала [a, b]
     */
    void set_interval_length(double length) {
        interval_length_ = std::abs(length);
    }
    
    /**
     * @brief Получение значения по ключу
     * @param key ключ
     * @return значение
     * @throws std::out_of_range если ключ не найден
     */
    Value get(const Key& key) {
        Key qkey = quantize_key(key);
        auto it = map_.find(qkey);
        if (it == map_.end()) {
            throw std::out_of_range("Key not found in cache");
        }
        // Перемещаем в конец (недавно использованный)
        move_to_back(it->second);
        return it->second->second;
    }
    
    /**
     * @brief Попытка получить значение (без исключения)
     * @param key ключ
     * @param value выходное значение
     * @return true если ключ найден, false иначе
     */
    bool try_get(const Key& key, Value& value) noexcept {
        try {
            Key qkey = quantize_key(key);
            auto it = map_.find(qkey);
            if (it != map_.end()) {
                move_to_back(it->second);
                value = it->second->second;
                return true;
            }
        } catch (...) {
            // Игнорируем ошибки
        }
        return false;
    }
    
    /**
     * @brief Добавление или обновление элемента
     * @param key ключ
     * @param value значение
     */
    void put(const Key& key, const Value& value) {
        Key qkey = quantize_key(key);
        auto it = map_.find(qkey);
        if (it != map_.end()) {
            // Обновляем существующий элемент
            it->second->second = value;
            move_to_back(it->second);
            return;
        }
        // Добавляем новый элемент
        items_.emplace_back(qkey, value);
        map_[qkey] = std::prev(items_.end());
        evict_if_needed();
    }
    
    /**
     * @brief Проверка наличия ключа
     * @param key ключ
     * @return true если ключ существует
     */
    bool contains(const Key& key) const noexcept {
        Key qkey = const_cast<LRUCache*>(this)->quantize_key(key);
        return map_.find(qkey) != map_.end();
    }
    
    /**
     * @brief Удаление элемента по ключу
     * @param key ключ
     * @return true если элемент был удалён
     */
    bool erase(const Key& key) {
        Key qkey = quantize_key(key);
        auto it = map_.find(qkey);
        if (it == map_.end()) return false;
        items_.erase(it->second);
        map_.erase(it);
        return true;
    }
    
    /**
     * @brief Очистка кэша
     */
    void clear() noexcept override {
        items_.clear();
        map_.clear();
    }
    
    /**
     * @brief Размер кэша
     */
    std::size_t size() const noexcept override {
        return items_.size();
    }
    
    /**
     * @brief Проверка на пустоту
     */
    bool empty() const noexcept override {
        return items_.empty();
    }
    
    /**
     * @brief Максимальный размер кэша
     */
    std::size_t max_size() const noexcept {
        return max_size_;
    }
    
    /**
     * @brief Установка максимального размера
     */
    void set_max_size(std::size_t size) {
        max_size_ = size;
        evict_if_needed();
    }
    
    /**
     * @brief Статистика использования
     */
    struct Stats {
        std::size_t hits = 0;
        std::size_t misses = 0;
        double hit_rate() const {
            std::size_t total = hits + misses;
            return total > 0 ? static_cast<double>(hits) / total : 0.0;
        }
    };
    
    mutable Stats stats;
    
    /**
     * @brief Безопасная версия get с подсчётом статистики
     */
    bool safe_get(const Key& key, Value& value) noexcept {
        if (try_get(key, value)) {
            ++stats.hits;
            return true;
        }
        ++stats.misses;
        return false;
    }
};

// Специализации для часто используемых типов

/**
 * @brief Кэш для значений полинома F(x)
 */
using EvaluationCache = LRUCache<double, double>;

/**
 * @brief Кэш для значений и производных {F, F', F''}
 */
using DerivativeCache = LRUCache<double, std::array<double, 3>>;

/**
 * @brief Кэш для базисных функций φₖ(x)
 */
using BasisCache = LRUCache<std::pair<double, std::size_t>, double>;

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_LRU_CACHE_H
